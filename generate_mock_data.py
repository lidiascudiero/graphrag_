import os

def create_mock_dataset():
    """Generates dummy ECSS documents to test the system without violating copyright."""
    os.makedirs("sample_data", exist_ok=True)
    
    mock_doc_1 = """ECSS-E-ST-40C - Software Engineering
1. Scope
This standard defines the principles for aerospace software engineering.

2. Software Unit Testing
2.1 The supplier [SHALL] perform unit testing on all software modules before integration.
2.2 The unit test plan [SHOULD] be reviewed by the quality assurance board.
2.3 The supplier [MAY] use automated testing tools.

3. Safety Critical Software
3.1 Safety critical software [SHALL] comply with the verification guidelines in ECSS-Q-ST-80C.
"""

    mock_doc_2 = """ECSS-Q-ST-80C - Software Product Assurance
1. Scope
This document outlines the quality assurance metrics for space systems.

2. Verification Guidelines
2.1 All test results [SHALL] be logged in the centralized repository.
2.2 The QA team [SHOULD] conduct audits every milestone.
"""

    with open("sample_data/ECSS-E-ST-40C_mock.txt", "w", encoding="utf-8") as f:
        f.write(mock_doc_1)
        
    with open("sample_data/ECSS-Q-ST-80C_mock.txt", "w", encoding="utf-8") as f:
        f.write(mock_doc_2)
        
    print(" Mock dataset generated successfully in the 'sample_data/' folder.")

if __name__ == "__main__":
    create_mock_dataset()
