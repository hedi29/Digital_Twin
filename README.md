# Digital Twin for Marketing Concept Evaluation

This project implements a **Digital Twin system** for evaluating marketing concepts using machine learning clustering and persona-based analysis. The system creates user personas from interaction data and predicts how different user segments would respond to new marketing concepts.

## 📋 Overview

The Digital Twin system uses **K-means clustering** to segment users into personas based on their interaction patterns, then evaluates marketing concepts against these personas to predict engagement and response rates.

### Key Features
- **Persona Creation**: Automatically clusters users into distinct personas using interaction data
- **Concept Evaluation**: Scores marketing concepts based on persona preferences
- **Demographic Alignment**: Considers age groups and user demographics in evaluation
- **Response Prediction**: Predicts "yes", "maybe", or "no" responses with reasoning

## 🚀 Environment Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd Spiking_Jelly
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**
   ```bash
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Running the Code

To run the example demonstration:

```bash
python Digital_Twin.py
```

