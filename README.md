# NumPy Training Module - Day 00 Piscine

This repository contains the solutions for the "Day 00 - NumPy" piscine, implemented as a single Python script, along with comprehensive audit suites to verify correctness, reproducibility, and compliance with strict coding constraints (e.g., no loops).

## 📂 Project Structure

- **`numpy_exercises.py`**: The main solution script containing functions for Exercises 00 through 09.
- **`audit_numpy.py`**: A V1 Test Suite using `unittest` and AST analysis to verify strict compliance (no loops) and expected outputs.
- **`audit_numpy_v2.py`**: A V2 Comprehensive Audit Suite covering deep validation of environment, types, memory layout, security, and performance.

## 🚀 Tasks & Solutions

The exercises cover fundamental to advanced NumPy concepts:

1.  **Ex 00: Environment**: Checking Python/NumPy versions.
2.  **Ex 01: Mixed-Type Array**: Handling object arrays.
3.  **Ex 02: Reshape**: Manipulating array dimensions.
4.  **Ex 03: Slicing**: Advanced slicing, filtering (odd/even), and masking. **(No Loops)**
5.  **Ex 04: Random**: Deterministic generation using seeds (Normal, Ints).
6.  **Ex 05: Concatenation**: Joining and reshaping arrays.
7.  **Ex 06: Broadcasting**: Frame pattern generation and broadcasting multiplication. **(No Loops)**
8.  **Ex 07: NaN Handling**: Conditional replacement using `np.where`. **(No Loops)**
9.  **Ex 08: Wine Quality**: Loading CSVs, optimizing types, and calculating statistics.
10. **Ex 09: Optimization**: Finding optimal pairings using `itertools` to minimize squared differences. **(No Loops)**

## 🛠 Usage & Verification

### Prerequisites
- Python 3.7+
- NumPy 1.18+

### Running the Solutions
To execute the solutions and view the output formats required for the audit:
```bash
python3 numpy_exercises.py
```

### Running the Audit Suites
To verify the solutions against all constraints and test cases:

**1. Basic Audit (V1)**
Checks for loops, outputs, and basic correctness.
```bash
python3 audit_numpy.py
```

**2. Comprehensive Audit (V2)**
Performs deep inspection including memory layout, security checks, and edge cases.
```bash
python3 audit_numpy_v2.py
```

## 📊 Test Case Coverage

The audit suites cover the following categories:
- **Environment**: Virtual env, Python/NumPy compatibility.
- **Types**: Mixed types, nested structures, boolean handling.
- **Memory**: Reshaping, views vs copies, strides.
- **Randomness**: Seed reproducibility, bounds checking.
- **Broadcasting**: Scalar, vector-to-matrix, and outer product broadcasting.
- **NaN Handling**: Propagation, vectorization, and dtype preservation.
- **Security**: AST scan for `eval`/`exec`.
- **Quality**: Function length and structure checks.

## 📝 Notes
- **Exercise 8 (Wine)**: The script checks for `winequality-red.csv`. If missing, it gracefully performs a simulated run printing the implemented logic.
- **Exercise 9 (Football)**: Returns the optimal pairing indices `[[0 1 2 3 5], [8 7 9 4 6]]` matching the provided audit logs.
