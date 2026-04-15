# AI Assistant Guidelines for ML Course Project

When assisting with this project, the AI must strictly adhere to the following rules:

### 1. Consult the Project Specification (`project_spec.md`)
Always check the `project_spec.md` file before proposing major changes or executing steps. The spec outlines the grading criteria, Kaggle submission limits, coding standards (no repetitive AI-generated code blocks), and specific rules about which models are permissible (e.g. feed-forward multi-layer perceptrons only for Neural Networks, all types of boosting are okay).

### 2. Verify Column Meanings (`dataset_description.md`)
**Never guess or assume** what a column or feature means. Aircraft terminology and strike logging contain specific domain nuances. Every time you are dealing with a new feature, you must cross-reference it against the `dataset_description.md` to ensure accurate domain knowledge and proper data cleaning/engineering decisions.

### 3. Check the Syllabus Limits (`syllabus.md`)
You are strictly restricted to the Machine Learning concepts covered in this class. Before suggesting an advanced preprocessing technique, dimension reduction algorithm, or model architecture, you must verify against the `syllabus.md` to ensure the technique is within the bounds of what the student has learned. Do not suggest algorithms or methods that bypass the class curriculum, as this violates the project rules.

---
*Note: The AI agent should review these documents actively whenever context is lost or when initiating a new major phase of the ML pipeline (EDA, Data Cleaning, Feature Engineering, Modeling).*
