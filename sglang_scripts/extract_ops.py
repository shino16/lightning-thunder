#!/usr/bin/env python3
"""
Script to extract operations from inductor_* classes in model graph files.

Usage:
    # Process all models and combine into one file
    python extract_ops.py

    # Process a single model
    python extract_ops.py "Qwen/Qwen3-0.6B"
"""

import sys
import os
import re
from pathlib import Path
from typing import Set, List, Dict, Tuple


def get_indentation(line: str) -> int:
    """Return the number of leading spaces in a line."""
    return len(line) - len(line.lstrip())


def normalize_operation(op: str) -> str:
    """
    Normalize an operation by replacing variable names with placeholders.
    This allows deduplication of operations that differ only in variable names.

    Examples:
        output_parallel.is_cpu -> VAR.is_cpu
        torch.ops.sglang.inplace_all_reduce(output_parallel, group_name = 'tp:0')
            -> torch.ops.sglang.inplace_all_reduce(VAR, group_name = 'tp:0')
    """
    # Pattern to match variable-like names (not module paths or keywords)
    # Match identifiers that could be local variables:
    # - Not preceded by 'torch.' or other module paths
    # - Typical variable names with underscores, alphanumeric

    # Replace variable names in common patterns:
    # 1. variable.method or variable.attribute at the start
    # 2. function(variable, ...)
    # 3. kwarg = variable

    # Start by identifying tokens
    normalized = op

    # Pattern 1: variable at the beginning followed by dot (like "output_parallel.is_cpu")
    # But preserve module paths like "torch.ops" or "torch.cuda"
    # Don't replace if it's a known module name
    known_modules = ['torch', 'numpy', 'np', 'F', 'nn', 'math']
    if not any(op.startswith(f'{mod}.') for mod in known_modules):
        normalized = re.sub(r'^([a-z_]\w*)\.', r'VAR.', normalized)

    # Pattern 2: Replace function arguments (variables between parentheses and commas)
    # But keep string literals, numbers, and module paths
    # Match variables as function arguments: func(var, var)
    def replace_arg(match):
        arg = match.group(1).strip()
        # Keep if it's: a number, a string, a module path (contains .), or a keyword
        if (arg.startswith(("'", '"')) or  # string
            arg.replace('.', '').replace('-', '').isdigit() or  # number
            '.' in arg or  # module path like torch.ops
            arg in ['True', 'False', 'None']):  # keywords
            return match.group(0)
        else:
            return 'VAR' + match.group(2)

    # Replace variables after opening parens or commas
    normalized = re.sub(r'([,(])\s*([a-z_]\w*)(\s*[,)])', lambda m: m.group(1) + 'VAR' + m.group(3), normalized)

    # Pattern 3: After = in keyword arguments (but preserve string values, numbers, module paths)
    def replace_kwarg_value(match):
        value = match.group(1).strip()
        # Keep string literals, numbers, module paths
        if (value.startswith(("'", '"')) or
            '.' in value or  # module path or attribute access
            value.replace('-', '').replace('.', '').isdigit() or
            value in ['True', 'False', 'None']):
            return match.group(0)
        else:
            return '= VAR'

    # This one is trickier - for now, skip normalizing kwarg values
    # since they might be meaningful (like group_name = 'tp:0')

    return normalized


def extract_ops_from_file(filepath: Path) -> Set[str]:
    """Extract operations from inductor_* classes in a single file."""
    ops = set()

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is an inductor_* class definition
        if re.match(r'^\s*class\s+inductor_\w+\(', line):
            class_indent = get_indentation(line)
            i += 1

            # Process lines in the class body
            while i < len(lines):
                current_line = lines[i]
                current_indent = get_indentation(current_line)

                # If we've dedented back to or past the class level, we're done with this class
                if current_line.strip() and current_indent <= class_indent:
                    break

                # Check if this is a nested class definition
                if re.match(r'^\s*class\s+\w+\(', current_line):
                    nested_class_indent = current_indent
                    # Skip the entire nested class
                    i += 1
                    while i < len(lines):
                        nested_line = lines[i]
                        nested_indent = get_indentation(nested_line)
                        if nested_line.strip() and nested_indent <= nested_class_indent:
                            # Don't increment i here, let the outer loop handle it
                            break
                        i += 1
                    # Continue without incrementing i again since we've already positioned correctly
                    continue

                # Process lines directly in the inductor_* class body
                stripped = current_line.strip()

                # Skip empty lines, comment-only lines, function definitions, and returns
                if not stripped or stripped.startswith('#') or stripped.startswith('def ') or stripped.startswith('return'):
                    i += 1
                    continue

                # Extract meaningful expressions from the line
                expression = None

                if '=' in stripped:
                    # Has assignment - extract right-hand side
                    eq_pos = stripped.find('=')

                    if ';' in stripped:
                        # Pattern: variable = expression; ...
                        semicolon_pos = stripped.find(';', eq_pos)
                        expression = stripped[eq_pos + 1:semicolon_pos].strip()
                    else:
                        # Pattern: variable = expression
                        expression = stripped[eq_pos + 1:].strip()
                else:
                    # No assignment - use the whole line as expression
                    if ';' in stripped:
                        # Pattern: expression; ...
                        semicolon_pos = stripped.find(';')
                        expression = stripped[:semicolon_pos].strip()
                    else:
                        # Pattern: expression
                        expression = stripped

                # Add non-empty expressions (skip trivial values like "None")
                if expression and expression not in ['None', '()', '{}', '[]']:
                    ops.add(expression)

                i += 1
        else:
            i += 1

    return ops


def extract_ops_from_model(model_name: str) -> List[str]:
    """Extract operations from all files for a given model."""
    # Get the base directory (project root)
    base_dir = Path(__file__).parent.parent
    source_dir = base_dir / 'gm' / model_name

    if not source_dir.exists():
        print(f"Error: Directory {source_dir} does not exist")
        sys.exit(1)

    # Find all Python files in the directory
    py_files = list(source_dir.glob('*.py'))

    if not py_files:
        print(f"Error: No Python files found in {source_dir}")
        sys.exit(1)

    print(f"Processing {len(py_files)} files from {source_dir}...")

    # Collect all operations with pattern-based deduplication
    # Map: normalized_pattern -> first occurrence of that pattern
    pattern_to_op: Dict[str, str] = {}

    for py_file in py_files:
        ops = extract_ops_from_file(py_file)

        # For each operation, check if we've seen this pattern before
        for op in ops:
            normalized = normalize_operation(op)
            if normalized not in pattern_to_op:
                # First time seeing this pattern, save the original operation
                pattern_to_op[normalized] = op

        print(f"  - {py_file.name}: found {len(ops)} operations")

    # Return the first occurrence of each unique pattern, sorted
    unique_ops = list(pattern_to_op.values())
    return sorted(unique_ops)


def save_ops(model_dir_name: str, ops: List[str], output_dir: Path):
    """Save extracted operations to the output file."""
    output_file = output_dir / f"{model_dir_name}.py"

    # Create parent directories if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write operations to file
    with open(output_file, 'w') as f:
        for op in ops:
            f.write(f"{op}\n")

    print(f"  → Saved {len(ops)} unique operations to {output_file}")


def find_all_models(gm_dir: Path) -> List[Tuple[str, Path]]:
    """
    Find all model directories in the gm/ directory.
    Returns list of (model_name, model_path) tuples.
    """
    models = []

    # Walk through all subdirectories
    for item in gm_dir.rglob('*'):
        if item.is_dir():
            # Check if this directory contains .py files (indicating it's a model directory)
            py_files = list(item.glob('*.py'))
            if py_files:
                # Use just the directory name (not full path) as model name
                model_name = item.name
                models.append((model_name, item))

    return sorted(models, key=lambda x: x[0])


def main():
    base_dir = Path(__file__).parent.parent  # Go up to project root
    gm_dir = base_dir / 'gm'

    if len(sys.argv) > 1:
        # Single model mode
        model_name = sys.argv[1]
        print(f"Processing single model: {model_name}")

        # Extract operations
        ops = extract_ops_from_model(model_name)

        # Save to output file (use just the last part of the path)
        model_dir_name = Path(model_name).name
        save_ops(model_dir_name, ops, gm_dir)
    else:
        # Process all models and combine into one file
        print("Scanning for all models in gm/ directory...")
        models = find_all_models(gm_dir)

        if not models:
            print("Error: No model directories found in gm/")
            sys.exit(1)

        print(f"\nFound {len(models)} model(s):")
        for model_name, model_path in models:
            print(f"  - {model_name} ({model_path.relative_to(base_dir)})")

        print("\n" + "=" * 70)
        print("Processing all models...")
        print("=" * 70)

        # Collect all operations across all models with global deduplication
        global_pattern_to_op: Dict[str, str] = {}

        for model_name, model_path in models:
            print(f"\n[{model_name}]")

            # Get relative path from gm/ for extraction
            rel_path = model_path.relative_to(gm_dir)

            try:
                # Extract operations from this model
                ops = extract_ops_from_model(str(rel_path))
                print(f"  Found {len(ops)} unique operations")

                # Add to global collection with pattern-based deduplication
                for op in ops:
                    normalized = normalize_operation(op)
                    if normalized not in global_pattern_to_op:
                        global_pattern_to_op[normalized] = op

            except Exception as e:
                print(f"  ✗ Error processing {model_name}: {e}")

        # Get all unique operations and sort
        all_unique_ops = sorted(global_pattern_to_op.values())

        # Save to single combined file
        output_file = gm_dir / "all_ops.py"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            for op in all_unique_ops:
                f.write(f"{op}\n")

        print("\n" + "=" * 70)
        print(f"Done! Saved {len(all_unique_ops)} unique operations to {output_file}")
        print("=" * 70)


if __name__ == '__main__':
    main()

