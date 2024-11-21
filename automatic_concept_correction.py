import torch
import json
from typing import List, Dict
def create_concept_class_membership(
        self,
        concepts: List[str],
        classes: List[str],
        acc_file_path: str
    ) -> torch.Tensor:
        """
        Create a boolean tensor representing concept-class relationships from a JSON file.
        
        Args:
            concepts: List of concept names
            classes: List of class names
            acc_file_path: Path to the JSON file containing concept-class associations
            
        Returns:
            torch.Tensor: Boolean tensor of shape (num_concepts, num_classes)
            
        Raises:
            FileNotFoundError: If acc_file_path doesn't exist
            json.JSONDecodeError: If JSON file is invalid
        """
        # Initialize the membership tensor with zeros
        concept_class_membership = torch.zeros(
            (len(concepts), len(classes)), 
            dtype=torch.bool
        )
        
        # Read and process the JSON file
        with open(acc_file_path, "r") as f:
            file_data = json.load(f)
            
            # Process concepts that apply to all classes
            if "ALL" in file_data:
                for concept in file_data["ALL"]:
                    if concept in concepts:
                        concept_index = concepts.index(concept)
                        concept_class_membership[concept_index, :] = True
            
            # Process class-specific concepts
            for class_index, class_name in enumerate(classes):
                if class_name in file_data:
                    for concept in file_data[class_name]:
                        if concept in concepts:
                            concept_index = concepts.index(concept)
                            concept_class_membership[concept_index][class_index] = True
                            
        return concept_class_membership