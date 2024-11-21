# test_automatic_concept_correction.py
import unittest
import tempfile
import os
import json
import torch
from automatic_concept_correction import create_concept_class_membership

class TestConceptMembership(unittest.TestCase):
    def setUp(self):
        self.concepts = ["fur", "wings", "tail", "beak"]
        self.classes = ["dog", "bird", "cat"]
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
        
    def test_basic_concept_assignment(self):
        # Create test JSON file
        test_data = {
            "dog": ["fur", "tail"],
            "bird": ["wings", "beak"],
            "cat": ["fur", "tail"]
        }
        
        test_file = os.path.join(self.temp_dir, "test_basic.json")
        with open(test_file, "w") as f:
            json.dump(test_data, f)
            
        result = create_concept_class_membership(
            self.concepts,
            self.classes,
            test_file
        )
        
        expected = torch.tensor([
            [True, False, True],  # fur
            [False, True, False], # wings
            [True, False, True],  # tail
            [False, True, False]  # beak
        ], dtype=torch.bool)
        
        self.assertTrue(torch.equal(result, expected))
        
    def test_all_concept_assignment(self):
        test_data = {
            "ALL": ["tail"],
            "dog": ["fur"],
            "bird": ["wings", "beak"],
            "cat": ["fur"]
        }
        
        test_file = os.path.join(self.temp_dir, "test_all.json")
        with open(test_file, "w") as f:
            json.dump(test_data, f)
            
        result = create_concept_class_membership(
            self.concepts,
            self.classes,
            test_file
        )
        
        expected = torch.tensor([
            [True, False, True],  # fur
            [False, True, False], # wings
            [True, True, True],   # tail (applies to all)
            [False, True, False]  # beak
        ], dtype=torch.bool)
        
        self.assertTrue(torch.equal(result, expected))
        
    def test_nonexistent_concept(self):
        test_data = {
            "dog": ["fur", "nonexistent_concept"],
            "bird": ["wings"],
            "cat": ["fur"]
        }
        
        test_file = os.path.join(self.temp_dir, "test_nonexistent.json")
        with open(test_file, "w") as f:
            json.dump(test_data, f)
            
        result = create_concept_class_membership(
            self.concepts,
            self.classes,
            test_file
        )
        
        expected = torch.tensor([
            [True, False, True],  # fur
            [False, True, False], # wings
            [False, False, False], # tail
            [False, False, False]  # beak
        ], dtype=torch.bool)
        
        self.assertTrue(torch.equal(result, expected))
        
    def test_empty_file(self):
        test_data = {}
        
        test_file = os.path.join(self.temp_dir, "test_empty.json")
        with open(test_file, "w") as f:
            json.dump(test_data, f)
            
        result = create_concept_class_membership(
            self.concepts,
            self.classes,
            test_file
        )
        
        expected = torch.zeros((len(self.concepts), len(self.classes)), dtype=torch.bool)
        self.assertTrue(torch.equal(result, expected))
        
    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            create_concept_class_membership(
                self.concepts,
                self.classes,
                "nonexistent_file.json"
            )

if __name__ == '__main__':
    unittest.main()