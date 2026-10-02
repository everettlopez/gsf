import json
from pathlib import Path


class EnvironmentTree:
    def __init__(self, locations: dict):
        self.locations = locations

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(data["locations"])

    def get_root_locations(self):
        return list(self.locations.keys())

    def get_children(self, location_name: str):
        node = self.find_node(location_name)
        
        if node is None:
            return []
        
        return list(node.keys())

    def find_node(self, target_name, current=None):
        if current is None:
            current = self.locations

        for name, node in current.items():
            if name == target_name:
                return node

            if isinstance(node, dict):
                result = self.find_node(target_name, node)

                if result is not None:
                    return result

        return None