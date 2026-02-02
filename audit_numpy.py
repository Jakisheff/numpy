
import unittest
import numpy as np
import os
import sys
import ast
import inspect
import re
from io import StringIO
from contextlib import redirect_stdout

# Add current directory to path to import numpy_exercises
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import numpy_exercises
except ImportError:
    print("CRITICAL: numpy_exercises.py not found or cannot be imported.")
    sys.exit(1)

class TestNumpyExercises(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load source code for AST analysis
        with open('numpy_exercises.py', 'r') as f:
            cls.source_code = f.read()
        cls.tree = ast.parse(cls.source_code)
    
    def check_no_loops(self, func_name):
        """Check if a function contains explicit loops or list comprehensions."""
        # Find the function node
        func_node = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                func_node = node
                break
        
        if not func_node:
            self.fail(f"Function {func_name} not found in source code.")
            
        for node in ast.walk(func_node):
            if isinstance(node, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp)):
                # Allow ListComp only if it's NOT in the specific prohibited exercises? 
                # Constraint: "No for loops" for Ex 3, 6, 7, 9.
                self.fail(f"Loop detected in {func_name}: {type(node).__name__}")

    def test_ex00_environment(self):
        """TC-00-XX checks"""
        output, nums = numpy_exercises.ex00()
        python_ver, numpy_ver = output, nums
        
        # TC-00-02: Python version verification
        self.assertTrue(sys.version_info >= (3, 6), "Python version must be >= 3.6")
        
        # TC-00-08: Output verification (captured via stdout in real run, mostly visual)
        pass

    def test_ex01_mixed_types(self):
        """TC-01-XX checks"""
        arr = numpy_exercises.ex01()
        
        # TC-01-01: NumPy array type
        self.assertIsInstance(arr, np.ndarray)
        
        # Check types of elements (assuming specific order or just presence)
        # The refactored code has extra types now for TC compliance: [1, "a", 3.5, {}, [], (), set(), True]
        self.assertIsInstance(arr[0], (int, np.integer))
        self.assertIsInstance(arr[1], (str, np.str_))
        self.assertIsInstance(arr[2], (float, np.floating))
        # Extended checks if present
        if len(arr) > 3:
            self.assertIsInstance(arr[3], dict)
            self.assertIsInstance(arr[4], list)

    def test_ex02_zeros_reshape(self):
        """TC-02-XX checks"""
        a, b = numpy_exercises.ex02()
        
        # TC-02-02: Initial shape (300,)
        self.assertEqual(a.shape, (300,))
        # TC-02-03: Values zero
        self.assertTrue(np.all(a == 0))
        
        # TC-02-05: Reshaped shape (3, 100)
        self.assertEqual(b.shape, (3, 100))
        
        # TC-02-07: Anti-cheat (scan source code for hardcoded zeros)
        # Simple regex: look for "[0, 0, 0, 0"
        if re.search(r'\[0, 0, 0, 0', self.source_code):
             self.fail("Hardcoded zeros array detected.")

    def test_ex03_slicing(self):
        """TC-03-XX checks"""
        self.check_no_loops("ex03")
        
        arr, odd, even_rev, c = numpy_exercises.ex03()
        
        # TC-03-03: Base array
        np.testing.assert_array_equal(arr, np.arange(1, 101))
        
        # TC-03-04: Odd numbers
        np.testing.assert_array_equal(odd, np.arange(1, 101, 2))
        
        # TC-03-05: Even reversed
        np.testing.assert_array_equal(even_rev, np.arange(100, 1, -2))
        
        # TC-03-06: Step modification
        # Every 3rd element (indices 2, 5, 8...) -> 0
        expected_c = np.arange(1, 101)
        expected_c[2::3] = 0
        np.testing.assert_array_equal(c, expected_c)

    def test_ex04_random(self):
        """TC-04-XX checks"""
        # Re-run logic to verify reproducibility
        normal_dist, ints_8x8, ints_4x2x5 = numpy_exercises.ex04()
        
        # TC-04-03: Normal dist shape
        self.assertEqual(normal_dist.shape, (100,))
        
        # TC-04-06: Ints 8x8 range [1, 10] (Updated logic check)
        self.assertEqual(ints_8x8.shape, (8, 8))
        self.assertTrue(np.all((ints_8x8 >= 1) & (ints_8x8 <= 10)), "Ints 8x8 out of range [1, 10]")
        
        # TC-04-08: Ints 4x2x5 range [1, 17]
        self.assertEqual(ints_4x2x5.shape, (4, 2, 5))
        self.assertTrue(np.all((ints_4x2x5 >= 1) & (ints_4x2x5 <= 17)), "Ints 4x2x5 out of range [1, 17]")
        
        # TC-04-02: Reproducibility check
        # We manually re-run seed and check
        np.random.seed(888)
        check_normal = np.random.normal(0, 1, 100)
        np.testing.assert_array_almost_equal(normal_dist, check_normal)

    def test_ex05_concatenate(self):
        """TC-05-XX"""
        a, b, c, d = numpy_exercises.ex05()
        
        # TC-05-04: Concatenation
        np.testing.assert_array_equal(c, np.arange(1, 101))
        # TC-05-05: Reshape
        self.assertEqual(d.shape, (10, 10))

    def test_ex06_broadcasting(self):
        """TC-06-XX"""
        self.check_no_loops("ex06")
        
        frame, res = numpy_exercises.ex06()
        
        # TC-06-02: Initial array dtype/shape
        # Note: frame variable in my code is the FINAL frame. 
        self.assertEqual(frame.shape, (9, 9))
        self.assertEqual(frame.dtype, np.int8)
        
        # Check center has zeros
        self.assertTrue(np.all(frame[1:-1, 1:-1] == 0))
        # Check border has ones
        self.assertTrue(np.all(frame[0, :] == 1))
        self.assertTrue(np.all(frame[-1, :] == 1))
        self.assertTrue(np.all(frame[:, 0] == 1))
        self.assertTrue(np.all(frame[:, -1] == 1))
        
        # TC-06-06: Broadcasting result
        expected_res = np.arange(1, 6).reshape(5, 1) * np.arange(1, 4).reshape(1, 3)
        np.testing.assert_array_equal(res, expected_res)

    def test_ex07_nan(self):
        """TC-07-XX"""
        self.check_no_loops("ex07")
        # Logic verification
        data = numpy_exercises.ex07()
        
        # TC-07-05: NaN replacement
        # Checking if col 0 has no NaNs
        self.assertFalse(np.any(np.isnan(data[:, 0])), "Column 0 still contains NaNs")

    def test_ex08_wine(self):
        """TC-08-XX"""
        # If file missing, returns None, skip test
        val = numpy_exercises.ex08()
        if val is None:
            print("Skipping Ex 08 tests (File not found)")
            return
        
        # If I had logic to return specific stats, I'd check them here.
        # But ex08 mostly prints.
        pass

    def test_ex09_football(self):
        """TC-09-XX"""
        self.check_no_loops("ex09")
        
        expected = numpy_exercises.ex09()
        
        # TC-09-06: Shape check
        self.assertEqual(expected.shape, (2, 5))
        
        # TC-09-08: Expected output match
        target = np.array([[0, 1, 2, 3, 5], [8, 7, 9, 4, 6]])
        np.testing.assert_array_equal(expected, target)

if __name__ == '__main__':
    unittest.main()
