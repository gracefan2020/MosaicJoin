import os
import json
import random
import pandas as pd
import numpy as np
import argparse
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sys

# Add parent directory to path to import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.models import GeminiLLM, OpenAILLM

load_dotenv('data-harmonization-agent/.env')

# --- CONFIGURATION ---
# Fixed models per provider (no overrides)
OPENAI_MODEL = "o4-mini"
GEMINI_MODEL = os.getenv("PORTKEY_MODEL", "@vertexai/gemini-2.5-pro")

INPUT_DIR = "/scratch/yl6624/data-harmonization/NYC_open_data/cleaned_csvs"
# Overlap Ratios
MIN_ROWS = 50
ROW_OVERLAP_RANGE = (0.1, 0.3)
COL_OVERLAP_RANGE = (0.2, 0.4)

MAX_TABLES = 10
MAX_WORKERS = 1

# Difficulty Distribution Strategy:
# - 20% L1 (exact match, no change)
# - 30% L2 (syntactic variation only)  
# - 50% L3-L4 (semantic rename and/or transformation)
# This is controlled by the renaming and transformation probabilities below

# Probability Gates
P_ACCEPT_LLM_RENAME = 0.90      # 90% use LLM synonym when in semantic mode

# Ambiguous Mapping Configuration
# Strategy: Ask LLM to analyze ACTUAL column pairs and find natural ambiguity
P_AMBIGUOUS_PAIR = 0.40  # 40% probability to KEEP when LLM finds feasible scenario
AMBIGUOUS_PER_PAIR = (1, 3)  # Max number of ambiguous columns per table pair

# False Positive Configuration (no-match columns that look like matches)
P_FALSE_POSITIVE = 0.30  # 30% probability to KEEP when LLM finds feasible scenario
FALSE_POSITIVE_PER_PAIR = (1, 2)  # Max number of false positive columns per table pair

# Global LLM provider (will be set by argument parser)
llm_provider = None
llm_client = None


class UnifiedLLM:
    """Unified interface for OpenAI and Gemini LLMs with call tracking."""
    
    def __init__(self, provider="openai"):
        """
        Initialize LLM provider.
        
        Args:
            provider: "openai" or "gemini"
        """
        self.provider = provider.lower()
        self.call_count = 0  # Track total number of LLM calls
        
        if self.provider == "openai":
            # Temperature in __init__ is not used (generate_json overrides it)
            # Set to 0.0 as placeholder since actual temperature is passed per-call
            self.llm = OpenAILLM(model_name=OPENAI_MODEL, temperature=0.0)
            self.is_openai = True
        elif self.provider == "gemini":
            # Temperature in __init__ is not used (generate_json overrides it)
            # Set to 0.0 as placeholder since actual temperature is passed per-call
            self.llm = GeminiLLM(model_name=GEMINI_MODEL, temperature=0.0)
            self.is_openai = False
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'gemini'")
    
    def reset_call_count(self):
        """Reset the call counter."""
        self.call_count = 0
    
    def get_call_count(self):
        """Get the current call count."""
        return self.call_count
    
    def generate_json(self, prompt, temperature=0.7):
        """
        Generate JSON response from prompt.
        
        Args:
            prompt: The prompt text
            temperature: Temperature for generation (0.0-1.0)
                         0.0-0.3: Precision/accuracy tasks
                         0.4-0.6: Balanced tasks
                         0.7-0.9: Creative tasks
                         
        Note: o4-mini only supports temperature=1.0 (default). 
              This method automatically uses 1.0 for o4-mini regardless of requested value.
        
        Returns:
            Parsed JSON dict or None on error
        """
        self.call_count += 1  # Increment call counter
        
        try:
            # OpenAI has native JSON mode support via response_format parameter
            # Gemini (via LangChain) doesn't have this, so we need to add JSON instruction to prompt
            if self.is_openai:
                # o4-mini only supports temperature=1.0 (default), so override for this model
                # For other OpenAI models, use the requested temperature
                actual_temperature = 1.0 if OPENAI_MODEL == "o4-mini" else temperature
                
                # Use OpenAI client directly with native JSON mode for guaranteed JSON output
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=actual_temperature,
                    response_format={"type": "json_object"}  # Native JSON mode - no prompt instruction needed
                )
                return json.loads(response.choices[0].message.content)
            else:
                # Gemini doesn't have native JSON mode, so we add instruction to prompt
                prompt_with_json = prompt + "\n\nIMPORTANT: Return your response as valid JSON only, with no additional text or explanation."
                
                messages = [{"role": "user", "content": prompt_with_json}]
                response_text = self.llm.generate(messages, temperature=temperature)
                
                # Try to parse JSON (may need to extract from markdown code blocks)
                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                return json.loads(response_text)
        except Exception as e:
            print(f"Warning: LLM generation failed: {e}")
            return None


class DifficultyClassifier:
    """Classifies column matches into difficulty levels for SCHEMA MATCHING."""
    
    @staticmethod
    def classify_match(original_col, new_col, has_transform, transform_type, is_ambiguous=False):
        """
        Classify difficulty level (L1-L4) based on match characteristics.
        This is specifically for SCHEMA MATCHING difficulty.
        
        Returns: dict with level, reasoning, and metadata
        """
        # L4 (Expert): Ambiguous mappings or complex semantic + value transform
        if is_ambiguous:
            return {
                "level": "L4",
                "reasoning": "Ambiguous mapping requiring value analysis to disambiguate",
                "requires_backtrack": True
            }
        
        # L1 (Easy): Exact name match (with or without transformation)
        # Schema matching is about finding the correspondence, not about value transformation
        # Value transformation difficulty is tracked separately in value_mapping_difficulty
        if original_col == new_col:
            return {
                "level": "L1",
                "reasoning": "Exact name match" + (" with value transformation" if has_transform else ", no transformation needed"),
                "requires_backtrack": False
            }
        
        # L2 (Medium): High name similarity with minor syntactic differences
        if not has_transform or transform_type in ["syntactic_only", "simple_format"]:
            # Calculate string similarity
            similarity = DifficultyClassifier._string_similarity(original_col, new_col)
            if similarity > 0.7:
                return {
                    "level": "L2",
                    "reasoning": "High name similarity with minor syntactic differences",
                    "requires_backtrack": False
                }
        
        # L3 (Hard): Semantic renaming or moderate value transformation
        if has_transform and transform_type in ["semantic_rename", "moderate_transform"]:
            return {
                "level": "L3",
                "reasoning": "Semantic renaming or moderate value transformation required",
                "requires_backtrack": False
            }
        
        # L4 (Expert): Complex transformation or very low similarity
        if has_transform and transform_type in ["complex_transform", "categorical_mapping"]:
            return {
                "level": "L4",
                "reasoning": "Complex value transformation or categorical mapping required",
                "requires_backtrack": False
            }
        
        # Default to L3
        return {
            "level": "L3",
            "reasoning": "Requires semantic matching and careful analysis",
            "requires_backtrack": False
        }
    
    @staticmethod
    def _string_similarity(s1, s2):
        """Calculate Jaccard similarity between two strings."""
        s1_tokens = set(s1.lower().replace('_', ' ').replace('-', ' ').split())
        s2_tokens = set(s2.lower().replace('_', ' ').replace('-', ' ').split())
        if not s1_tokens or not s2_tokens:
            return 0.0
        intersection = len(s1_tokens & s2_tokens)
        union = len(s1_tokens | s2_tokens)
        return intersection / union if union > 0 else 0.0


class ValueMappingClassifier:
    """Classifies value transformations into difficulty levels for VALUE MAPPING."""
    
    @staticmethod
    def classify_value_mapping(transform_info, col_type):
        """
        Classify value mapping difficulty (VM1-VM5) based on transformation complexity.
        
        Args:
            transform_info: dict with transformation details including code and operations
            col_type: detected column type
            
        Returns: dict with level, reasoning, and metadata
        """
        if not transform_info:
            return {
                "level": "VM1",
                "reasoning": "No value transformation needed",
                "num_operations": 0
            }
        
        num_ops = transform_info.get('num_operations', 1)
        complexity = transform_info.get('complexity', 'simple_format')
        has_conditional = transform_info.get('has_conditional_logic', False)
        operations = transform_info.get('operations', [])
        expert_type = transform_info.get('expert_type', None)
        has_multi_lambda = transform_info.get('multi_lambda', None) is not None
        has_value_dict = transform_info.get('value_mapping_dict', None) is not None
        
        # VM5 (Master): Expert-level transformations with multi-lambda or dictionary learning
        if complexity in ["expert_multi_lambda", "expert_dictionary"]:
            if has_multi_lambda:
                return {
                    "level": "VM5",
                    "reasoning": "Multi-lambda pipeline: Sequential transformations chained together, each building on previous step",
                    "num_operations": num_ops,
                    "expert_type": "multi_lambda"
                }
            elif has_value_dict:
                return {
                    "level": "VM5",
                    "reasoning": "Dictionary learning: Explicit value mapping dictionary inferred from data patterns",
                    "num_operations": num_ops,
                    "expert_type": "dictionary_learning"
                }
            else:
                # Fallback if marked as expert but missing specific markers
                return {
                    "level": "VM5",
                    "reasoning": "Expert-level transformation requiring advanced techniques",
                    "num_operations": num_ops,
                    "expert_type": expert_type or "unknown"
                }
        
        # VM1 (Easy): Single simple operation, no conditionals
        # This includes: case changes, strip, simple arithmetic (x*100), rounding, basic casting
        if num_ops == 1 and complexity in ["simple_format", "syntactic_only"] and not has_conditional:
            return {
                "level": "VM1",
                "reasoning": "Single simple format operation (e.g., case change, whitespace trim, multiply by constant)",
                "num_operations": num_ops
            }
        
        # ALSO VM1: Single operation that's just multiplication/rounding (even if moderate_transform)
        # Common case: x * 100 for percentage conversion
        if num_ops == 1 and not has_conditional and complexity == "moderate_transform":
            # Check if it's really just simple arithmetic
            code = transform_info.get('transform_code', '')
            if code:
                # Simple patterns that should be VM1
                simple_patterns = [
                    '* 100',  # multiply by constant
                    'round(',  # just rounding
                    '.strip()',  # just strip
                    '.upper()',  # just case
                    '.lower()',  # just case
                    '.title()',  # just case
                ]
                if any(pattern in code for pattern in simple_patterns) and 'if' not in code:
                    return {
                        "level": "VM1",
                        "reasoning": "Single simple arithmetic or format operation",
                        "num_operations": num_ops
                    }
        
        # VM2 (Medium): 2 simple operations OR single moderate operation with complexity
        if (num_ops == 2 and complexity in ["simple_format", "moderate_transform"] and not has_conditional):
            return {
                "level": "VM2",
                "reasoning": "Two simple operations or moderate transformation (e.g., strip + case, unit conversion + round)",
                "num_operations": num_ops
            }
        
        # Also VM2: Single operation but with some complexity (substring, replace, split)
        if (num_ops == 1 and complexity == "moderate_transform" and not has_conditional):
            return {
                "level": "VM2",
                "reasoning": "Moderate transformation operation (e.g., substring extraction, replace, split)",
                "num_operations": num_ops
            }
        
        # VM3 (Hard): Conditional logic OR 3-4 operations OR complex single operation
        if (num_ops == 1 and complexity in ["complex_transform", "categorical_mapping"]) or \
           (num_ops >= 3 and num_ops <= 4) or \
           (has_conditional and num_ops <= 2):
            return {
                "level": "VM3",
                "reasoning": "Complex transformation requiring conditional logic or multiple dependent operations",
                "num_operations": num_ops
            }
        
        # VM4 (Expert): Multiple complex operations OR heavy conditional logic
        if num_ops >= 5 or \
           (has_conditional and num_ops >= 3) or \
           (complexity == "categorical_mapping" and has_conditional):
            return {
                "level": "VM4",
                "reasoning": "Multiple complex transformations with extensive conditional logic (e.g., multi-step categorical mappings, value-dependent operations)",
                "num_operations": num_ops
            }
        
        # Default to VM2
        return {
            "level": "VM2",
            "reasoning": "Moderate value transformation complexity",
            "num_operations": num_ops
        }


class ValueTransformEngine:
    """Generates diverse, realistic value transformations with LLM-driven difficulty selection."""
    
    @staticmethod
    def detect_column_type(series):
        """Detect column type: date, numeric, categorical, text."""
        sample = series.dropna().head(20)
        if len(sample) == 0:
            return "text"
        
        # Try date detection (suppress warnings for format inference)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, message='.*Could not infer format.*')
                pd.to_datetime(sample, errors='raise')
            return "date"
        except:
            pass
        
        # Numeric detection
        if pd.api.types.is_numeric_dtype(series):
            try:
                unique_ratio = series.nunique() / len(series)
                if unique_ratio < 0.1:  # Low unique ratio suggests categorical
                    return "categorical_numeric"
                return "numeric"
            except (TypeError, ValueError):
                # Series contains unhashable types (e.g., dicts, lists from bad transformations)
                # Fall back to text type
                return "text"
        
        # Categorical vs text
        try:
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.3:  # Categorical
                return "categorical"
        except (TypeError, ValueError):
            # Series contains unhashable types - check what type of values we have
            sample_values = series.dropna().head(5)
            if len(sample_values) > 0:
                first_val = sample_values.iloc[0]
                if isinstance(first_val, (list, tuple, dict)):
                    # Transformation returned non-scalar values - treat as text
                    return "text"
        
        return "text"
    
    @staticmethod
    def generate_transformation(col_name, col_type, sample_values):
        """
        Generate a transformation with LLM choosing the appropriate difficulty level.
        The LLM analyzes the column and decides which difficulty is most appropriate,
        then generates the transformation - all in ONE call.
        
        Args:
            col_name: column name
            col_type: detected column type
            sample_values: sample values from the column
            
        Returns: dict with transformation details or None if generation fails
        """
        sample_str = ', '.join([str(v) for v in sample_values[:10]])
        
        prompt = f"""
You are a data transformation expert. Analyze this column and generate an appropriate transformation.

Column: {col_name}
Type: {col_type}
Sample values: {sample_str}

**Your task**:
1. Analyze the column to determine what difficulty level is most appropriate
2. Generate a transformation at that difficulty level
3. Return the transformation code and metadata

**Difficulty Levels** (choose the most appropriate):

**simple** (25% preferred): Basic format changes
- Case changes, whitespace trim, simple casting, basic prefix/suffix
- Examples: "lambda x: str(x).upper()", "lambda x: str(x).strip()"
- Use when: Column needs only minor formatting adjustments

**moderate** (25% preferred): Unit conversions, substring operations, format changes
- Unit conversions (ONLY if name suggests units), substring extraction, basic mappings
- Examples: "lambda x: str(x).split('-')[0]", "lambda x: str(x).replace('STU-', '')"
- Use when: Column needs meaningful but straightforward transformations

**complex** (20% preferred): Conditional logic, value-dependent transformations
- Single conditional with different transformations per branch
- Examples: "lambda x: str(x).upper() if len(str(x)) > 5 else str(x).lower()"
- Use when: Column values need different handling based on conditions

**multi_step** (15% preferred): Multiple operations for different value subsets
- 2-5 different operations with conditions
- Examples: "lambda x: 'M' if x in ['M','m','Male'] else ('F' if x in ['F','f','Female'] else x)"
- Use when: Column has rich, varied data that can be meaningfully subdivided

**expert** (15% preferred): L5/VM5 - ADVANCED TRANSFORMATIONS (NEW!)
Two types of expert transformations:

**Type A - Multi-Lambda Pipeline**: ONLY when transformations CANNOT be elegantly chained
**CRITICAL**: This is NOT for simple method chaining like x.strip().lower().title()
**ONLY USE when you need:**
1. **Intermediate result inspection**: Result of step N determines what step N+1 should be
2. **Complex parsing with state**: Extract multiple parts, then recombine based on what was found
3. **Try-except across steps**: Step 1 tries something, step 2 handles failure differently
4. **Type-dependent logic**: Parse → validate type → transform based on type
5. **List/dict operations**: Split into list → filter/map based on content → rejoin

**GOOD Multi-Lambda Examples** (TRULY can't be one lambda):
```
Example 1: Date parsing with fallback formats
  Step 1: Try parse as "YYYY-MM-DD", if fails return None
  Step 2: If step 1 failed, try parse as "MM/DD/YYYY", else keep result
  Step 3: Extract year only if parsing succeeded, else "INVALID"
  → Can't chain: Each step must inspect previous result

Example 2: Conditional ID restructuring  
  Step 1: Split by delimiter, store as list
  Step 2: If list len > 3, keep first 3, ELSE reverse the list
  Step 3: Join with "-" if was >3, else join with "_"
  → Can't chain: Step 2 decision affects step 3 logic

Example 3: Extract-validate-transform
  Step 1: Extract numeric portion using regex
  Step 2: If numeric > 1000, divide by 1000, else multiply by 100
  Step 3: Format with "K" if divided, else "x100"
  → Can't chain: Step 2 affects step 3 formatting
```

**BAD Multi-Lambda Examples** (DON'T use expert - use simpler level):
```
x.strip().lower().title() - simple chaining, use moderate
x.replace('A', 'B').upper() - simple chaining, use simple
x.split('-')[0].strip() - simple chaining, use moderate
Single conditional - use complex level instead
```

**Type B - Dictionary Learning**: Infer value mappings from sample data
- Analyze sample values to discover categorical patterns
- Build explicit mapping dictionary based on observed patterns
- **ONLY USE when**: 
  1. You see 3-10 distinct categorical values in samples
  2. Values follow clear categorical structure (gender, status, codes, booleans)
  3. Multiple variants map to same concept (M/m/Male → Male)
- Examples:
  * Gender: {{"M": "Male", "F": "Female", "m": "Male", "f": "Female"}}
  * Status codes: {{"A": "Active", "I": "Inactive", "P": "Pending"}}
  * Boolean: {{"Y": "Yes", "N": "No", "1": "Yes", "0": "No"}}

**CRITICAL RULES**:
1. Choose the difficulty that makes sense for THIS column (not random)
2. For multi_lambda: ONLY use if transformations truly can't be chained elegantly
3. If you can write it as "lambda x: x.method1().method2().method3()", DO NOT use expert
4. Transformations MUST preserve data semantics
5. Be realistic - would this transformation occur in real data integration?

**Return JSON**:
{{
    "difficulty_chosen": "simple|moderate|complex|multi_step|expert",
    "difficulty_reasoning": "Why you chose this difficulty for this column",
    
    // For simple/moderate/complex/multi_step: Use this field
    "transform_code": "lambda x: <python code>",
    
    // For expert Type A (multi-lambda): Use these fields
    "expert_type": "multi_lambda|dictionary_learning",  // Only for expert difficulty
    "multi_lambda": [  // Only for expert + multi_lambda
        {{"step": 1, "code": "lambda x: <transformation>", 
          "description": "What this step does", 
          "output_example": "What this produces"}},
        {{"step": 2, "code": "lambda x: <depends on step 1>", 
          "description": "How this depends on step 1 result", 
          "output_example": "..."}}
    ],
    "why_cant_chain": "Explain why this CANNOT be written as one chained lambda",
    
    // For expert Type B (dictionary learning): Use these fields
    "value_mapping_dict": {{  // Only for expert + dictionary_learning
        "source_value": "target_value", ...
    }},
    "transform_code": "lambda x: mapping_dict.get(str(x), x)",
    
    // Common fields
    "description": "What the transformation does",
    "complexity": "simple_format|moderate_transform|complex_transform|categorical_mapping|expert_multi_lambda|expert_dictionary",
    "has_conditional_logic": true/false,
    "num_operations": <number>,
    "operations": [{{"condition": "...", "transform": "...", "description": "..."}}],
    "example_input": "sample value",
    "example_output": "transformed value"
}}

**Guidelines by type**:
- **date**: Format changes. Expert multi-lambda: Try multiple parsers with fallback
- **numeric**: Unit conversions. Expert multi-lambda: Extract → validate range → apply conditional scaling
- **categorical**: Value mappings. Expert dictionary: BEST USE CASE - learn from samples
- **text**: String operations. Expert multi-lambda: Parse → inspect → conditional reconstruction
- **identifiers**: Extract components. Expert multi-lambda: Split → validate format → transform based on findings

**When to choose EXPERT multi-lambda**:
YES: Intermediate result determines next step's logic
YES: Need try-except with different handling per step
YES: Parse → inspect → conditional transform pattern

NO: Can be written as x.method1().method2()
NO: Simple sequential operations without dependencies
NO: Single conditional (use complex level)

**Expert Level Examples**:

**Multi-Lambda Example** (Date with fallback):
```json
{{
    "difficulty_chosen": "expert",
    "expert_type": "multi_lambda",
    "multi_lambda": [
        {{"step": 1, "code": "lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce') if x else pd.NaT", 
          "description": "Try ISO format", "output_example": "2023-01-15 or NaT"}},
        {{"step": 2, "code": "lambda x: pd.to_datetime(x, format='%m/%d/%Y', errors='coerce') if pd.isna(x) else x", 
          "description": "If failed, try US format", "output_example": "2023-01-15"}},
        {{"step": 3, "code": "lambda x: x.year if not pd.isna(x) else 'INVALID'", 
          "description": "Extract year if succeeded", "output_example": "2023 or 'INVALID'"}}
    ],
    "why_cant_chain": "Step 2 only executes if step 1 failed (pd.isna check), step 3 behavior depends on success",
    "num_operations": 3,
    "complexity": "expert_multi_lambda"
}}
```

**Dictionary Example**:
```json
{{
    "difficulty_chosen": "expert",
    "expert_type": "dictionary_learning",
    "value_mapping_dict": {{
        "M": "Male", "m": "Male", "F": "Female", "f": "Female"
    }},
    "transform_code": "lambda x: mapping_dict.get(str(x), 'Unknown')",
    "num_operations": 1,
    "complexity": "expert_dictionary"
}}
```

**BE VERY STRICT**: If you can write as single lambda with method chaining, do NOT use expert multi-lambda.

Choose wisely!
"""
        
        try:
            # Temperature 0.7: Balance creativity with consistency
            result = llm_client.generate_json(prompt, temperature=0.7)
            if result is None:
                return None
            
            # Ensure num_operations is set
            if 'num_operations' not in result:
                result['num_operations'] = len(result.get('operations', []))
                if result['num_operations'] == 0:
                    result['num_operations'] = 1
            
            return result
        except Exception as e:
            print(f"Warning: Transform generation failed: {e}")
            return None
    
    @staticmethod
    def get_transformation_or_identity(col_name, col_type, sample_values):
        """
        Main entry point: Let LLM choose difficulty and generate transformation in one call.
        Returns identity mapping (None) if generation fails.
        
        Returns: dict with transformation details or None for identity mapping
        """
        # Single LLM call: LLM chooses difficulty AND generates transformation
        transform_result = ValueTransformEngine.generate_transformation(
            col_name, col_type, sample_values
        )
        
        # If generation fails, return identity mapping (safe fallback)
        if transform_result is None:
            return None
        
        return transform_result


class AmbiguousMapper:
    """
    Creates intentionally ambiguous mappings that require value analysis to resolve.
    These will be used to generate backtracking trajectories.
    """
    
    @staticmethod
    def create_ambiguous_pair(df, source_cols, target_cols, existing_overlaps):
        """
        Create an ambiguous column pair using a practical approach:
        - Find columns with similar names in target
        - Create ambiguous source name based on actual column names
        - Ensure values can distinguish them
        
        Returns: dict with ambiguous mapping config or None
        """
        # Find candidates: columns in target but not in existing overlaps
        available_target = [c for c in target_cols if c not in existing_overlaps and c in df.columns]
        
        if len(available_target) < 2:
            return None
        
        # Strategy: Find pairs of columns with related names or semantics
        # Ask LLM to analyze actual column names and find ambiguous opportunities
        
        # Sample 5 random column pairs to find the best ambiguous scenario
        sample_pairs = []
        for _ in range(min(5, len(available_target) // 2)):
            pair = random.sample(available_target, 2)
            if pair[0] in df.columns and pair[1] in df.columns:
                sample_pairs.append(pair)
        
        if not sample_pairs:
            return None
        
        # Get column info for better LLM analysis
        pair_info = []
        for pair in sample_pairs:
            info = {
                "col1": pair[0],
                "col1_samples": df[pair[0]].dropna().head(5).tolist(),
                "col1_type": str(df[pair[0]].dtype),
                "col2": pair[1],
                "col2_samples": df[pair[1]].dropna().head(5).tolist(),
                "col2_type": str(df[pair[1]].dtype)
            }
            pair_info.append(info)
        
        prompt = f"""
You are analyzing real table columns to find opportunities for creating ambiguous schema matching scenarios.

I have these column pairs from the target table:
{json.dumps(pair_info, indent=2, default=str)}

**Task**: Find ONE pair that can create a good ambiguous mapping scenario, where:

1. **The columns are related but distinct** (e.g., both dates, both IDs, both measurements)
2. **You can create a SHORT, ambiguous source name** that could plausibly match EITHER column
3. **The values clearly distinguish them** (different types, ranges, or semantics)

**Good Examples**:
- "birth_date" vs "admission_date" → ambiguous source: "date" or "patient_date"
- "total_amount" vs "discount_amount" → ambiguous source: "amount"  
- "home_address" vs "work_address" → ambiguous source: "address"
- "patient_id" vs "visit_id" → ambiguous source: "id"

**Return JSON**:
{{
    "feasible": true/false,
    "correct_target": "actual column name from the pair",
    "trap_target": "other column name from the pair",
    "ambiguous_source_name": "SHORT name (1-2 words) that fits both",
    "trap_appeal": "why trap looks plausible (be specific about naming)",
    "value_distinguisher": "how values differ (be specific about the actual sample values)"
}}

**RULES**:
1. Pick the pair where columns are MOST related semantically
2. The ambiguous_source_name should be a SUBSTRING or SIMPLIFICATION of both column names
3. **CRITICAL**: The ambiguous_source_name MUST be DIFFERENT from both column names (no exact matches!)
4. Only return feasible=true if you can create a genuinely confusing scenario
5. If no good pair exists, return feasible=false

Be picky! Only generate if there's a real opportunity for confusion based on actual column names.
"""
        
        try:
            # Temperature 0.6: Needs creativity for finding ambiguous scenarios,
            # but also accuracy to identify genuinely confusing pairs
            result = llm_client.generate_json(prompt, temperature=0.6)
            if result is None:
                return None
            
            if result.get("feasible"):
                # Validate that both columns exist
                correct = result["correct_target"]
                trap = result["trap_target"]
                ambiguous_name = result["ambiguous_source_name"]
                
                # CRITICAL VALIDATION: Ambiguous name must be DIFFERENT from both targets
                # Otherwise it's not ambiguous - it's just an exact match!
                if ambiguous_name == correct or ambiguous_name == trap:
                    return None  # Reject this scenario
                
                if correct in df.columns and trap in df.columns:
                    return {
                        "source_column": ambiguous_name,
                        "wrong_target": trap,
                        "correct_target": correct,
                        "wrong_target_reason": result["trap_appeal"],
                        "discovery_reason": result["value_distinguisher"],
                        "name_similarity_explanation": f"'{ambiguous_name}' is ambiguous between '{correct}' and '{trap}'"
                    }
            return None
            
        except Exception as e:
            print(f"Warning: Ambiguous mapping generation failed: {e}")
            return None
    
    @staticmethod
    def should_use_for_backtracking(ambig_config, source_df, target_df):
        """
        Validate that the ambiguous mapping is suitable for backtracking.
        Checks that values actually differ in a meaningful way.
        
        Returns: (is_valid, reason)
        """
        source_col = ambig_config['source_column']
        correct_target = ambig_config['correct_target']
        trap_target = ambig_config['trap_target']
        
        if source_col not in source_df.columns:
            return False, "Source column not found"
        if correct_target not in target_df.columns or trap_target not in target_df.columns:
            return False, "Target columns not found"
        
        # Check that values are meaningfully different
        correct_vals = target_df[correct_target].dropna().head(10)
        trap_vals = target_df[trap_target].dropna().head(10)
        
        if len(correct_vals) == 0 or len(trap_vals) == 0:
            return False, "Insufficient values"
        
        # Simple heuristic: check if types differ or value ranges differ
        correct_type = ValueTransformEngine.detect_column_type(target_df[correct_target])
        trap_type = ValueTransformEngine.detect_column_type(target_df[trap_target])
        
        if correct_type != trap_type:
            return True, f"Different types: {correct_type} vs {trap_type}"
        
        # If same type, values should still be distinguishable
        return True, "Values differ meaningfully"
    
    @staticmethod
    def create_false_positive(df, source_unique_cols, target_cols):
        """
        Create a false positive scenario: a source column that looks like it should match
        a target column based on naming, but actually doesn't (and has no other match).
        
        This tests the agent's ability to recognize non-matches by examining values.
        
        Args:
            df: original dataframe
            source_unique_cols: columns that will be unique to source (not overlapping)
            target_cols: all target columns
            
        Returns: dict with false positive config or None
        """
        # Find candidates: source-unique columns that could be renamed to look like target columns
        if len(source_unique_cols) < 1 or len(target_cols) < 1:
            return None
        
        # Sample a few source-unique columns and target columns for LLM analysis
        sample_source = random.sample(source_unique_cols, min(3, len(source_unique_cols)))
        sample_target = random.sample(target_cols, min(5, len(target_cols)))
        
        # Get column info
        source_info = []
        for col in sample_source:
            if col in df.columns:
                info = {
                    "col": col,
                    "samples": df[col].dropna().head(5).tolist(),
                    "type": str(df[col].dtype)
                }
                source_info.append(info)
        
        target_info = []
        for col in sample_target:
            if col in df.columns:
                info = {
                    "col": col,
                    "samples": df[col].dropna().head(5).tolist(),
                    "type": str(df[col].dtype)
                }
                target_info.append(info)
        
        if not source_info or not target_info:
            return None
        
        prompt = f"""
You are analyzing columns to create a "false positive" scenario for schema matching.

**Goal**: Find a source column (that will NOT have a match in target) and a target column where:
1. You can rename the source column to look SIMILAR to the target column name
2. The renamed source looks like it SHOULD match the target based on naming
3. BUT the values are clearly DIFFERENT (different semantics, types, or ranges)
4. This tests if agents examine values, not just names

**Source columns (will be unique to source, no match in target)**:
{json.dumps(source_info, indent=2, default=str)}

**Target columns**:
{json.dumps(target_info, indent=2, default=str)}

**Good Examples**:
- Source "building_age" (values: 5, 12, 45) → rename to "age" 
  Target "age" (values: 25, 34, 52) → looks like match but different semantics (building vs person)
  
- Source "employee_count" (values: 100, 250, 500) → rename to "count"
  Target "visit_count" (values: 5, 12, 8) → similar naming but different scales/semantics
  
- Source "zip_code" (values: 10001, 10002) → rename to "code"
  Target "product_code" (values: "A123", "B456") → similar name but different types

**Return JSON**:
{{
    "feasible": true/false,
    "source_column": "original source column name",
    "renamed_source": "new name that looks similar to target",
    "target_column": "target column that looks like a match",
    "naming_similarity": "why the renamed source looks like it should match the target",
    "value_difference": "how values clearly differ (be specific about sample values)"
}}

**RULES**:
1. The renamed source should look plausibly related to the target column name
2. Values must clearly differ (different types, ranges, semantics, or units)
3. The rename should preserve the source column's semantic meaning
4. Only return feasible=true if you can create a genuinely confusing false positive
5. If no good scenario exists, return feasible=false

Be selective! Only generate if there's a real opportunity for confusion.
"""
        
        try:
            result = llm_client.generate_json(prompt, temperature=0.6)
            if result is None:
                return None
            
            if result.get("feasible"):
                source_col = result["source_column"]
                renamed_source = result["renamed_source"]
                target_col = result["target_column"]
                
                # Validate columns exist
                if source_col not in df.columns or target_col not in df.columns:
                    return None
                
                # Validate that renamed source is different from target
                # (we want them similar but not identical)
                if renamed_source == target_col:
                    return None
                
                return {
                    "source_column": source_col,
                    "renamed_source": renamed_source,
                    "target_column": target_col,
                    "naming_similarity": result["naming_similarity"],
                    "value_difference": result["value_difference"]
                }
            return None
            
        except Exception as e:
            print(f"Warning: False positive generation failed: {e}")
            return None


class MultiLambdaExecutor:
    """Helper class to execute multi-lambda pipelines and dictionary-based transformations."""
    
    @staticmethod
    def apply_multi_lambda_pipeline(series, lambda_steps):
        """
        Apply a sequence of lambda functions in order.
        Each lambda operates on the output of the previous lambda.
        
        Args:
            series: pandas Series to transform
            lambda_steps: list of dicts with 'code' and 'description' for each step
            
        Returns: transformed pandas Series
        """
        import pandas as pd
        import numpy as np
        
        result = series.copy()
        
        for step_info in lambda_steps:
            code = step_info.get('code', '')
            if not code:
                continue
                
            try:
                # Evaluate the lambda function
                if code.strip().startswith('lambda'):
                    func = eval(code, {"pd": pd, "np": np, "str": str, "float": float, "int": int})
                else:
                    # Wrap in lambda if needed
                    func = eval(f"lambda x: {code}", {"pd": pd, "np": np, "str": str, "float": float, "int": int})
                
                # Apply to the series
                def safe_apply(x):
                    try:
                        if pd.isna(x):
                            return x
                        return func(x)
                    except (AttributeError, TypeError, ValueError):
                        return x
                
                result = result.apply(safe_apply)
                
            except Exception as e:
                print(f"Warning: Multi-lambda step failed: {e}. Skipping step.")
                # Continue with next step using current result
                continue
        
        return result
    
    @staticmethod
    def apply_dictionary_mapping(series, mapping_dict, default_value=None):
        """
        Apply a dictionary-based value mapping.
        
        Args:
            series: pandas Series to transform
            mapping_dict: dict mapping source values to target values
            default_value: value to use when key not in dict (None = keep original)
            
        Returns: transformed pandas Series
        """
        import pandas as pd
        
        def safe_map(x):
            try:
                if pd.isna(x):
                    return x
                # Try exact match first
                if x in mapping_dict:
                    return mapping_dict[x]
                # Try string version
                str_x = str(x)
                if str_x in mapping_dict:
                    return mapping_dict[str_x]
                # Return default or original
                return default_value if default_value is not None else x
            except:
                return x
        
        return series.apply(safe_map)
    
    @staticmethod
    def generate_combined_code(lambda_steps):
        """
        Generate a single combined lambda expression from multiple steps.
        This is for documentation purposes - shows the logical equivalent.
        
        Args:
            lambda_steps: list of lambda step dicts
            
        Returns: string representing combined lambda (may be complex)
        """
        if not lambda_steps:
            return "lambda x: x"
        
        if len(lambda_steps) == 1:
            return lambda_steps[0].get('code', 'lambda x: x')
        
        # For multiple steps, create nested lambda calls
        # This shows the conceptual equivalent but may be hard to read
        codes = [step.get('code', 'lambda x: x') for step in lambda_steps]
        
        # Build from inside out
        result = "x"
        for i, code in enumerate(codes):
            # Extract the expression part from "lambda x: <expression>"
            if 'lambda x:' in code:
                expr = code.split('lambda x:', 1)[1].strip()
                # Replace x with current result
                result = f"({expr.replace('x', result)})"
            else:
                result = f"({code.replace('x', result)})"
        
        return f"lambda x: {result}"


class ChaosEngine:
    @staticmethod
    def add_column_noise(col_name):
        """
        Generate syntactic variations for L2 difficulty.
        Creates realistic database naming variations.
        """
        choice = random.random()
        
        # 30% keep original (for L1 cases that fall through)
        if choice < 0.30:
            return col_name
        
        # 40% add common suffixes/prefixes
        elif choice < 0.70:
            variations = [
                f"{col_name}_src",
                f"{col_name}_val", 
                f"{col_name}_1",
                f"{col_name}_x",
                f"src_{col_name}",
                f"tbl_{col_name}",
                f"{col_name}_id",
                f"{col_name}_code",
                f"{col_name}_new",
            ]
            return random.choice(variations)
        
        # 20% case changes
        elif choice < 0.85:
            if col_name.islower():
                return col_name.upper()
            elif col_name.isupper():
                return col_name.lower()
            else:
                # Convert snake_case to camelCase or vice versa
                if '_' in col_name:
                    parts = col_name.split('_')
                    return parts[0] + ''.join(p.capitalize() for p in parts[1:])
                else:
                    return col_name.lower()
        
        # 10% remove characters (abbreviations)
        else:
            if len(col_name) > 5 and '_' in col_name:
                # Remove vowels from last part
                parts = col_name.split('_')
                last = parts[-1]
                abbrev = ''.join([c for c in last if c not in 'aeiouAEIOU'])
                if abbrev:
                    parts[-1] = abbrev
                    return '_'.join(parts)
            return col_name

    @staticmethod
    def apply_value_code(series, code_str):
        """Safely apply transformation code to series with proper error handling."""
        try:
            import pandas as pd
            import numpy as np
            
            # Handle both lambda functions and plain code
            # If it's already a lambda, use it directly
            if code_str.strip().startswith('lambda'):
                func = eval(code_str, {"pd": pd, "np": np, "str": str, "float": float, "int": int})
            else:
                # If it's plain code like "x.lower()", wrap it in a lambda
                func = eval(f"lambda x: {code_str}", {"pd": pd, "np": np, "str": str, "float": float, "int": int})
            
            # Apply function with proper NaN handling and type validation
            def safe_apply(x):
                try:
                    # Handle NaN values - return as-is
                    if pd.isna(x):
                        return x
                    # Apply the transformation
                    result = func(x)
                    
                    # CRITICAL: Ensure result is a scalar value (not list, tuple, dict)
                    # Non-scalar values cause issues with pandas operations like nunique()
                    if isinstance(result, (list, tuple, dict)):
                        # Non-scalar result - convert to string or return original
                        # For lists/tuples, try to join if all elements are strings
                        if isinstance(result, (list, tuple)) and len(result) > 0:
                            try:
                                if all(isinstance(item, str) for item in result):
                                    return ', '.join(result)
                            except:
                                pass
                        # Fallback: return original value if transformation produces non-scalar
                        return x
                    
                    # Ensure result is not NaN (unless input was NaN)
                    return result
                except (AttributeError, TypeError, ValueError) as e:
                    # Common errors: trying to call string methods on non-strings,
                    # or operations on NaN values
                    # Return original value on error
                    return x
            
            return series.apply(safe_apply)
        except Exception as e:
            # If the function itself can't be created, return original series
            # Only print warning if it's not a common type error (to reduce noise)
            error_str = str(e)
            if not any(err in error_str.lower() for err in ['attribute', 'type', 'split', 'iterable', 'nan']):
                print(f"Warning: Transform failed: {e}")
            return series


class SyntheticGenerator:
    def __init__(self, output_base):
        self.output_base = output_base
        os.makedirs(output_base, exist_ok=True)

    def get_architect_plan(self, df, filename):
        """Pass 1: Just identify the critical Join Keys."""
        sample = df.sample(min(5, len(df))).to_json(orient='records')
        columns = list(df.columns)
        
        prompt = f"""
        Table '{filename}' columns: {columns}. Sample: {sample}
        Identify 'Semantic Keys' (IDs, Names, composite keys) that MUST overlap to allow joining.
        Return JSON: {{ "semantic_overlap_cols": ["col1", "col2"] }}
        """
        try:
            # Temperature 0.2: Low temperature for precision task (identifying keys)
            result = llm_client.generate_json(prompt, temperature=0.2)
            return result
        except:
            return None

    def get_semantic_enrichment(self, df, overlap_cols):
        """
        Pass 2: The 'Hard Mode' Generator.
        Forces LLM to generate synonyms/transforms for the SPECIFIC overlap list.
        """
        sample = df[overlap_cols].sample(min(3, len(df))).to_json(orient='records')
        
        prompt = f"""
        I have two tables sharing these columns: {overlap_cols}.
        Here is a sample of the data: {sample}

        Task: Create a 'Source Table' schema that is semantically different but equivalent.
        For EVERY column in the list:
        1. Generate a 'hard' synonym (e.g., 'DOB' for 'BirthDate', 'Boro' for 'Borough').
        2. Create a Python value transformation code (e.g., "x.lower()", "x.split('-')[0]", "str(x) + '_v'"). 
           - Be creative. If numeric, maybe cast to string or add unit. 
           - If categorical, map values.
           - If ID, remove prefix.
           - IMPORTANT: Keep transformations semantically meaningful (don't scale age by 100!)

        Return JSON:
        {{
            "enrichments": {{
                "original_col_name": {{
                    "new_name": "HardSynonym",
                    "transform_code": "x.lower()" 
                }}
            }}
        }}
        """
        try:
            # Temperature 0.7: High creativity for generating synonyms and transformations
            result = llm_client.generate_json(prompt, temperature=0.7)
            return result
        except:
            return None

    def process_table(self, file_path):
        filename = os.path.basename(file_path).replace(".csv", "")
        
        # Reset call counter for this table
        llm_client.reset_call_count()
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except:
            return

        if df.shape[0] < MIN_ROWS: return

        # --- STEP 1: Identify Keys ---
        plan = self.get_architect_plan(df, filename)
        if not plan: return

        # --- STEP 2: Determine Overlap Strategy ---
        semantic_overlap = [c for c in plan.get("semantic_overlap_cols", []) if c in df.columns]
        
        total_cols = len(df.columns)
        target_overlap_count = int(total_cols * random.uniform(*COL_OVERLAP_RANGE))
        needed_random = max(0, target_overlap_count - len(semantic_overlap))
        
        remaining_cols = [c for c in df.columns if c not in semantic_overlap]
        random.shuffle(remaining_cols)
        
        random_overlap = remaining_cols[:needed_random]
        
        # The Master List of Overlapping Columns
        final_overlap_cols = semantic_overlap + random_overlap
        
        # Split remaining unique columns
        pure_remainder = remaining_cols[needed_random:]
        split_ratio = random.uniform(0.3, 0.6)
        split_idx = int(len(pure_remainder) * split_ratio)
        
        source_unique = pure_remainder[:split_idx]
        target_unique = pure_remainder[split_idx:]
        
        source_cols = final_overlap_cols + source_unique
        target_cols = final_overlap_cols + target_unique

        # --- STEP 2.5: Generate Ambiguous Mappings ---
        ambiguous_mappings = []
        ambiguous_sources_to_create = {}
        ambiguous_found = 0
        ambiguous_kept = 0
        
        if len(target_cols) >= 2:
            max_attempts = 3
            
            for attempt in range(max_attempts):
                ambig_config = AmbiguousMapper.create_ambiguous_pair(
                    df, source_cols, target_cols, final_overlap_cols
                )
                
                if ambig_config:
                    ambiguous_found += 1
                    if random.random() < P_AMBIGUOUS_PAIR:
                        ambiguous_kept += 1
                        ambiguous_mappings.append(ambig_config)
                        ambiguous_sources_to_create[ambig_config['source_column']] = ambig_config['correct_target']
                        
                        wrong_target = ambig_config['wrong_target']
                        correct_target = ambig_config['correct_target']
                        
                        if wrong_target not in target_cols:
                            target_cols.append(wrong_target)
                        if correct_target not in target_cols:
                            target_cols.append(correct_target)
                        
                        if correct_target not in final_overlap_cols:
                            final_overlap_cols.append(correct_target)
                            if correct_target in source_unique:
                                source_unique.remove(correct_target)
                            if correct_target in target_unique:
                                target_unique.remove(correct_target)
                        
                        if len(ambiguous_mappings) >= max(AMBIGUOUS_PER_PAIR):
                            break
                else:
                    break

        # --- STEP 2.6: Generate False Positive Scenarios ---
        false_positive_scenarios = []
        false_positive_renames = {}  # source_col -> renamed_source
        false_positive_found = 0
        false_positive_kept = 0
        
        if len(source_unique) >= 1 and len(target_cols) >= 1:
            max_attempts = 3
            
            for attempt in range(max_attempts):
                # Only consider source columns not already used for other purposes
                available_source_unique = [c for c in source_unique 
                                          if c not in ambiguous_sources_to_create.values()
                                          and c not in false_positive_renames]
                
                if not available_source_unique:
                    break
                
                fp_config = AmbiguousMapper.create_false_positive(
                    df, available_source_unique, target_cols
                )
                
                if fp_config:
                    false_positive_found += 1
                    if random.random() < P_FALSE_POSITIVE:
                        false_positive_kept += 1
                        false_positive_scenarios.append(fp_config)
                        false_positive_renames[fp_config['source_column']] = fp_config['renamed_source']
                        
                        if len(false_positive_scenarios) >= max(FALSE_POSITIVE_PER_PAIR):
                            break
                else:
                    break

        # --- STEP 3: Semantic Enrichment ---
        enrichments = {}
        if final_overlap_cols:
            llm_response = self.get_semantic_enrichment(df, final_overlap_cols)
            if llm_response:
                enrichments = llm_response.get("enrichments", {})

        # --- STEP 4: Row Splitting ---
        row_overlap_pct = random.uniform(*ROW_OVERLAP_RANGE)
        n_overlap = int(len(df) * row_overlap_pct)
        if n_overlap < 5: n_overlap = 5
        
        indices = df.index.tolist()
        random.shuffle(indices)
        idx_overlap = indices[:n_overlap]
        remaining = indices[n_overlap:]
        cut = int(len(remaining) * 0.5)
        
        df_source = df.loc[idx_overlap + remaining[:cut], source_cols].copy()
        df_target = df.loc[idx_overlap + remaining[cut:], target_cols].copy()

        # --- STEP 5: Apply Perturbations with Dual Difficulty Classification ---
        ground_truth_matches = []
        val_trans_log = {}
        new_source_headers = {}
        
        # First, handle ambiguous columns
        for ambig_source_name, correct_target in ambiguous_sources_to_create.items():
            if correct_target in df.columns:
                df_source[ambig_source_name] = df.loc[df_source.index, correct_target]
                
                ambig_config = next(a for a in ambiguous_mappings if a['source_column'] == ambig_source_name)
                
                match_record = {
                    "source_column": ambig_source_name,
                    "target_column": correct_target,
                    "confidence": 1.0,
                    "match_type": "ambiguous_llm",
                    "schema_difficulty": "L4",
                    "schema_difficulty_reasoning": "Ambiguous mapping requiring value analysis to disambiguate",
                    "value_mapping_difficulty": "VM1",
                    "value_mapping_reasoning": "No value transformation (identity mapping)",
                    "value_mapping_num_operations": 0,
                    "transformation_code": None,
                    "ambiguous_config": {
                        "trap_target": ambig_config['wrong_target'],
                        "trap_appeal": ambig_config['wrong_target_reason'],
                        "value_distinguisher": ambig_config['discovery_reason'],
                        "name_similarity_explanation": ambig_config.get('name_similarity_explanation', ''),
                        "requires_backtrack": True
                    }
                }
                ground_truth_matches.append(match_record)

        for col in source_cols:
            
            # Skip if this was an ambiguous column we already handled
            if col in ambiguous_sources_to_create.values():
                continue
            
            ambig_config = next((a for a in ambiguous_mappings if a['source_column'] == col), None)
            
            # CASE A: Overlapping Column
            if col in final_overlap_cols:
                final_name = col
                match_type = "exact"
                has_transform = False
                transform_type = "syntactic_only"
                transform_code = None
                transform_info = None
                
                # Check for LLM Enrichment
                enrich_data = enrichments.get(col, {})
                llm_name = enrich_data.get("new_name")
                llm_code = enrich_data.get("transform_code")

                # 1. Apply Renaming Strategy (for Schema Matching difficulty)
                # 20% L1 (no change), 30% L2 (syntactic), 50% L3+ (semantic/transform)
                rename_strategy = random.random()
                
                if rename_strategy < 0.20:
                    # L1: Keep exact name
                    final_name = col
                    match_type = "exact"
                elif rename_strategy < 0.50:
                    # L2: Syntactic noise only
                    noisy_name = ChaosEngine.add_column_noise(col)
                    if noisy_name != col:
                        final_name = noisy_name
                        match_type = "syntactic_noise"
                    else:
                        final_name = col
                        match_type = "exact"
                else:
                    # L3+: Try LLM semantic renaming
                    if llm_name and random.random() < P_ACCEPT_LLM_RENAME:
                        final_name = llm_name
                        match_type = "semantic_llm"
                    else:
                        # Fallback to syntactic noise
                        noisy_name = ChaosEngine.add_column_noise(col)
                        if noisy_name != col:
                            final_name = noisy_name
                            match_type = "syntactic_noise"
                        else:
                            final_name = col
                            match_type = "exact"

                new_source_headers[col] = final_name

                # 2. Apply Value Transformation (LLM-driven)
                # 30% no transform, 70% attempt transform
                should_attempt_transform = random.random() < 0.70
                
                if should_attempt_transform:
                    # Safely detect column type (may fail if previous transformation returned unhashable types)
                    try:
                        col_type = ValueTransformEngine.detect_column_type(df_source[col])
                    except (TypeError, ValueError):
                        # Fallback if detect_column_type fails
                        col_type = "text"
                    sample_vals = df_source[col].dropna().head(20).tolist()
                    
                    # Check if using LLM enrichment code first (if available)
                    use_enrichment = llm_code and random.random() < 0.3  # 30% chance to use enrichment if available
                    
                    if use_enrichment:
                        # Use the enrichment code from semantic enrichment phase
                        df_source[col] = ChaosEngine.apply_value_code(df_source[col], llm_code)
                        transform_code = llm_code
                        transform_info = {
                            'complexity': 'moderate_transform',
                            'num_operations': 1,
                            'has_conditional_logic': False
                        }
                        has_transform = True
                        val_trans_log[final_name] = {
                            "original_col": col,
                            "code": llm_code,
                            "description": "LLM enrichment transformation",
                            "type": col_type,
                            "complexity": "moderate_transform",
                            "num_operations": 1
                        }
                    else:
                        # Use LLM-driven feasibility assessment and generation
                        # This will return None (identity mapping) if all generation fails
                        llm_transform = ValueTransformEngine.get_transformation_or_identity(
                            col, col_type, sample_vals
                        )
                        
                        if llm_transform:
                            # Check if this is an expert-level transformation
                            expert_type = llm_transform.get('expert_type', None)
                            difficulty_chosen = llm_transform.get('difficulty_chosen', '')
                            
                            if difficulty_chosen == 'expert' and expert_type:
                                # Handle expert-level transformations
                                if expert_type == 'multi_lambda':
                                    # Apply multi-lambda pipeline
                                    multi_lambda = llm_transform.get('multi_lambda', [])
                                    if multi_lambda:
                                        df_source[col] = MultiLambdaExecutor.apply_multi_lambda_pipeline(
                                            df_source[col], multi_lambda
                                        )
                                        # Generate combined code for documentation
                                        transform_code = MultiLambdaExecutor.generate_combined_code(multi_lambda)
                                        transform_info = {
                                            'complexity': 'expert_multi_lambda',
                                            'num_operations': len(multi_lambda),
                                            'has_conditional_logic': llm_transform.get('has_conditional_logic', False),
                                            'operations': llm_transform.get('operations', []),
                                            'expert_type': 'multi_lambda',
                                            'multi_lambda': multi_lambda
                                        }
                                        has_transform = True
                                        val_trans_log[final_name] = {
                                            "original_col": col,
                                            "code": transform_code,
                                            "multi_lambda": multi_lambda,
                                            "description": llm_transform.get('description', 'Multi-lambda pipeline transformation'),
                                            "type": col_type,
                                            "complexity": "expert_multi_lambda",
                                            "num_operations": len(multi_lambda),
                                            "expert_type": "multi_lambda"
                                        }
                                    else:
                                        # Fallback if multi_lambda is empty
                                        transform_info = None
                                        has_transform = False
                                
                                elif expert_type == 'dictionary_learning':
                                    # Apply dictionary-based mapping
                                    mapping_dict = llm_transform.get('value_mapping_dict', {})
                                    if mapping_dict:
                                        df_source[col] = MultiLambdaExecutor.apply_dictionary_mapping(
                                            df_source[col], mapping_dict
                                        )
                                        transform_code = llm_transform.get('transform_code')
                                        transform_info = {
                                            'complexity': 'expert_dictionary',
                                            'num_operations': 1,
                                            'has_conditional_logic': False,
                                            'operations': llm_transform.get('operations', []),
                                            'expert_type': 'dictionary_learning',
                                            'value_mapping_dict': mapping_dict
                                        }
                                        has_transform = True
                                        val_trans_log[final_name] = {
                                            "original_col": col,
                                            "code": transform_code,
                                            "value_mapping_dict": mapping_dict,
                                            "description": llm_transform.get('description', 'Dictionary-based value mapping'),
                                            "type": col_type,
                                            "complexity": "expert_dictionary",
                                            "num_operations": 1,
                                            "expert_type": "dictionary_learning"
                                        }
                                    else:
                                        # Fallback if dictionary is empty
                                        transform_info = None
                                        has_transform = False
                                else:
                                    # Unknown expert type, treat as regular
                                    transform_code = llm_transform.get('combined_code') or llm_transform.get('transform_code')
                                    if transform_code:
                                        df_source[col] = ChaosEngine.apply_value_code(df_source[col], transform_code)
                                        transform_info = {
                                            'complexity': llm_transform.get('complexity', 'moderate_transform'),
                                            'num_operations': llm_transform.get('num_operations', 1),
                                            'has_conditional_logic': llm_transform.get('has_conditional_logic', False),
                                            'operations': llm_transform.get('operations', [])
                                        }
                                        has_transform = True
                                        val_trans_log[final_name] = {
                                            "original_col": col,
                                            "code": transform_code,
                                            "description": llm_transform.get('description', 'LLM-generated transformation'),
                                            "type": col_type,
                                            "complexity": transform_info['complexity'],
                                            "num_operations": transform_info['num_operations'],
                                            "has_conditional_logic": transform_info.get('has_conditional_logic', False),
                                            "operations": transform_info.get('operations', [])
                                        }
                                    else:
                                        transform_info = None
                                        has_transform = False
                            else:
                                # Regular (non-expert) transformation
                                transform_code = llm_transform.get('combined_code') or llm_transform.get('transform_code')
                                if transform_code:
                                    df_source[col] = ChaosEngine.apply_value_code(df_source[col], transform_code)
                                    transform_info = {
                                        'complexity': llm_transform.get('complexity', 'moderate_transform'),
                                        'num_operations': llm_transform.get('num_operations', 1),
                                        'has_conditional_logic': llm_transform.get('has_conditional_logic', False),
                                        'operations': llm_transform.get('operations', [])
                                    }
                                    has_transform = True
                                    val_trans_log[final_name] = {
                                        "original_col": col,
                                        "code": transform_code,
                                        "description": llm_transform.get('description', 'LLM-generated transformation'),
                                        "type": col_type,
                                        "complexity": transform_info['complexity'],
                                        "num_operations": transform_info['num_operations'],
                                        "has_conditional_logic": transform_info.get('has_conditional_logic', False),
                                        "operations": transform_info.get('operations', [])
                                    }
                        # else: llm_transform is None, meaning identity mapping (no transformation)
                        # This is the safe fallback - values remain unchanged
                
                # 3. Dual Difficulty Classification
                schema_difficulty = DifficultyClassifier.classify_match(
                    original_col=col,
                    new_col=final_name,
                    has_transform=has_transform,
                    transform_type=transform_type,
                    is_ambiguous=(ambig_config is not None)
                )
                
                # Safely detect column type (may fail if transformation returned unhashable types)
                try:
                    detected_type = ValueTransformEngine.detect_column_type(df_source[col]) if col in df_source.columns else "text"
                except (TypeError, ValueError):
                    # Fallback if detect_column_type fails (e.g., unhashable types in series)
                    detected_type = "text"
                
                value_difficulty = ValueMappingClassifier.classify_value_mapping(
                    transform_info=transform_info,
                    col_type=detected_type
                )
                
                # Add value_mapping_difficulty to val_trans_log if exists
                if final_name in val_trans_log:
                    val_trans_log[final_name]["value_mapping_difficulty"] = value_difficulty['level']
                    val_trans_log[final_name]["value_mapping_reasoning"] = value_difficulty['reasoning']
                
                # Record Match
                match_record = {
                    "source_column": final_name,
                    "target_column": col,
                    "confidence": 1.0,
                    "match_type": match_type,
                    "schema_difficulty": schema_difficulty['level'],
                    "schema_difficulty_reasoning": schema_difficulty['reasoning'],
                    "value_mapping_difficulty": value_difficulty['level'],
                    "value_mapping_reasoning": value_difficulty['reasoning'],
                    "value_mapping_num_operations": value_difficulty['num_operations'],
                    "transformation_code": transform_code if has_transform else None
                }
                
                # Add expert-level metadata if applicable
                if transform_info and transform_info.get('expert_type'):
                    expert_type = transform_info['expert_type']
                    match_record["expert_transformation"] = {
                        "type": expert_type
                    }
                    
                    if expert_type == 'multi_lambda':
                        match_record["expert_transformation"]["multi_lambda"] = transform_info.get('multi_lambda', [])
                        match_record["expert_transformation"]["num_steps"] = len(transform_info.get('multi_lambda', []))
                    elif expert_type == 'dictionary_learning':
                        match_record["expert_transformation"]["value_mapping_dict"] = transform_info.get('value_mapping_dict', {})
                        match_record["expert_transformation"]["num_mappings"] = len(transform_info.get('value_mapping_dict', {}))
                
                if ambig_config:
                    match_record["ambiguous_config"] = {
                        "wrong_target": ambig_config['wrong_target'],
                        "wrong_target_reason": ambig_config['wrong_target_reason'],
                        "discovery_reason": ambig_config['discovery_reason'],
                        "requires_backtrack": True
                    }
                
                ground_truth_matches.append(match_record)

            # CASE B: Unique Column
            else:
                # Check if this is a false positive column that should be renamed
                if col in false_positive_renames:
                    new_source_headers[col] = false_positive_renames[col]
                else:
                    new_source_headers[col] = col 

        # Finalize
        df_source.rename(columns=new_source_headers, inplace=True)

        # Shuffle columns to make schema matching more challenging
        # This prevents agents from relying on column order
        source_columns = list(df_source.columns)
        target_columns = list(df_target.columns)
        random.shuffle(source_columns)
        random.shuffle(target_columns)
        df_source = df_source[source_columns]
        df_target = df_target[target_columns]

        # Save
        task_dir = os.path.join(self.output_base, filename)
        os.makedirs(task_dir, exist_ok=True)
        
        df_source.to_csv(os.path.join(task_dir, "source_table.csv"), index=False)
        df_target.to_csv(os.path.join(task_dir, "target_table.csv"), index=False)
        
        # Enhanced metadata
        schema_difficulty_dist = {}
        value_difficulty_dist = {}
        for match in ground_truth_matches:
            schema_level = match.get('schema_difficulty', 'L3')
            value_level = match.get('value_mapping_difficulty', 'VM1')
            schema_difficulty_dist[schema_level] = schema_difficulty_dist.get(schema_level, 0) + 1
            value_difficulty_dist[value_level] = value_difficulty_dist.get(value_level, 0) + 1
        
        # Get total LLM calls for this table
        total_llm_calls = llm_client.get_call_count()
        
        meta = {
            "original_table": filename,
            "source_dims": list(df_source.shape),
            "target_dims": list(df_target.shape),
            "overlap_rows": len(idx_overlap),
            "overlap_cols": list(final_overlap_cols),
            "schema_difficulty_distribution": schema_difficulty_dist,
            "value_mapping_difficulty_distribution": value_difficulty_dist,
            "has_ambiguous_mappings": len(ambiguous_mappings) > 0,
            "num_ambiguous": len(ambiguous_mappings),
            "has_false_positives": len(false_positive_scenarios) > 0,
            "num_false_positives": len(false_positive_scenarios),
            "llm_calls": total_llm_calls
        }
        
        full_json = {
            "meta": meta,
            "matches": ground_truth_matches,
            "value_transformations": val_trans_log,
            "ambiguous_scenarios": ambiguous_mappings if ambiguous_mappings else [],
            "false_positive_scenarios": false_positive_scenarios if false_positive_scenarios else []
        }
        
        with open(os.path.join(task_dir, "ground_truth.json"), "w") as f:
            json.dump(full_json, f, indent=2)
        
        # Enhanced logging
        ambig_info = ""
        if ambiguous_found > 0:
            ambig_info = f", Ambiguous: found={ambiguous_found} kept={ambiguous_kept}"
        
        fp_info = ""
        if false_positive_found > 0:
            fp_info = f", FalsePos: found={false_positive_found} kept={false_positive_kept}"
        
        print(f"✓ {filename}: {len(ground_truth_matches)} matches, Schema: {schema_difficulty_dist}, Value: {value_difficulty_dist}{ambig_info}{fp_info}, LLM calls: {total_llm_calls}")


def run_pipeline(provider="openai"):
    """Run the data generation pipeline.
    
    Args:
        provider: "openai" or "gemini"
    """
    global llm_provider, llm_client
    
    # Set global LLM client (uses fixed models per provider)
    llm_provider = provider
    llm_client = UnifiedLLM(provider=provider)
    
    # Determine output directory based on provider
    if provider == "gemini":
        # Clean model name for directory (remove @vertexai/ prefix and replace - with _)
        model_suffix = GEMINI_MODEL.replace("@vertexai/", "").replace("-", "_")
        output_dir = f"/scratch/yl6624/data-harmonization/data-harmonization-agent/data/nyc_open_data_{model_suffix}"
    else:
        output_dir = f"/scratch/yl6624/data-harmonization/data-harmonization-agent/data/nyc_open_data_{OPENAI_MODEL}"
    
    print(f"--- Using {provider.upper()} provider with model: {llm_client.llm.model_name} ---")
    print(f"--- Output directory: {output_dir} ---")
    
    generator = SyntheticGenerator(output_dir)
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    random.shuffle(input_files)
    
    input_files = input_files[:MAX_TABLES]
    print(f"--- Processing {len(input_files)} tables (Max Tables: {MAX_TABLES}) ---")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(tqdm(executor.map(generator.process_table, input_files), total=len(input_files)))


import glob
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data harmonization tasks")
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="LLM provider to use: 'openai' (uses o4-mini) or 'gemini' (uses PORTKEY_MODEL env var, default: @vertexai/gemini-2.5-pro)"
    )
    
    args = parser.parse_args()
    run_pipeline(provider=args.provider)