import json
import nbformat

# Replace with actual filename if different
notebook_path = "USA_1_GNN_Pilot.ipynb"

# Load the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

# Remove broken metadata
if "widgets" in nb.metadata:
    del nb.metadata["widgets"]
    print("Removed 'metadata.widgets'")

# Save the fixed notebook
fixed_path = notebook_path.replace(".ipynb", ".ipynb")
with open(fixed_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook cleaned and saved as: {fixed_path}")
