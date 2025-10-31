
# Jumbled Frames Reconstruction Challenge  


Reconstructs a **10-second, 1080p, 30 FPS video** whose **300 frames** have been randomly shuffled.  
The task is to restore the correct frame order using **AI-based similarity analysis**, **optimization**, and **parallel processing** for accurate and efficient reconstruction.



##  **Overview**

This project automatically reconstructs the correct temporal order of a video whose frames are jumbled.  
It leverages **deep feature embeddings** from a pre-trained CNN (ResNet50) and **structural similarity** (SSIM) to compute inter-frame relationships and determine the most likely chronological order.

The approach combines **AI-driven visual understanding** with **classical optimization**, ensuring high accuracy and speed even for long single-shot videos.



##  **Key Features**

- **AI-powered frame ordering** using deep feature embeddings (ResNet50)  
- **Perceptual similarity** with Structural Similarity Index (SSIM)  
- **Hybrid Greedy + Local Optimization** algorithm for precise sequencing  
- **Automatic direction detection** (handles reversed videos)  
- **Parallelized computation** using `ThreadPoolExecutor` for faster performance  
- **Works on any jumbled single-shot video** (no scene cuts required)  



##  **How to Run the Project**

### **1. Clone the Repository**
Open your terminal and clone this GitHub repository to your local machine:

```bash
git clone https://github.com/MarmikKaila/Jumbled-Frames-Reconstruction-Challenge.git
cd Jumbled-Frames-Reconstruction-Challenge
```

### **2️. Create the requirements.txt File**
Create a new file named `requirements.txt` in the project folder and paste the following lines into it. This file lists all the required libraries.
```
opencv-python
tensorflow
numpy
scikit-image
scikit-learn
tqdm
```

### **3️. Create & Activate the Virtual Environment**

```bash
# Create the virtual environment
python -m venv .venv

# Activate the environment
# On Windows (PowerShell/CMD):
.\.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```
`(You should see (.venv) appear at the beginning of your terminal prompt.)`

### **4️. Install Dependencies**
While inside the activated virtual environment, install all the libraries from your `requirements.txt` file:

```bash
pip install -r requirements.txt
```
### **5. Add Your Jumbled Video**
Place your video in the same folder as the `reconstruct.py` script and `rename` it to::

```
jumbled_video.mp4
```
### **6. Run the Reconstruction Script**
Execute the main Python script:

```bash
python reconstruct.py
```
### **7. Check the Output**
After successful execution, a new video will be saved as:

```
reconstructed_video.mp4
```

 ##  **Jumbled Video**
 ```
https://drive.google.com/file/d/1DbR9yap-vgUaPiI3hCEKUnniXr-TrdOt/view
```

  ##  **Reconstructed Video**
```
https://drive.google.com/file/d/16qNp4ceUp0Mc4r5eZlgsO6hoG9q5uKRN/view?usp=drive_link
```



## **Algorithm Summary**

### **Feature Extraction**
- Uses a **pre-trained ResNet50** model (ImageNet weights) to extract deep feature embeddings from each frame.  
- These embeddings capture high-level scene understanding (objects, motion, lighting).

### **Low-Level Structure Features**
- Converts frames to grayscale and computes **SSIM** values for local pixel-based similarity.  
- This helps distinguish frames that look semantically similar but differ structurally.

### **Similarity Computation**
- Calculates a **hybrid similarity score** combining:
  - Cosine similarity between feature vectors  
  - SSIM between corresponding grayscale frames  
- Final similarity = `0.7 * DeepFeatureSim + 0.3 * SSIM`

### **Frame Ordering (Greedy Heuristic)**
- Begins with a random frame  
- Iteratively selects the **next most similar frame** using the hybrid similarity metric  
- Continues until all frames are sequenced

### **Local Optimization (2-Opt Refinement)**
- Refines ordering by swapping neighboring frames if doing so reduces total dissimilarity  
- Inspired by the **Travelling Salesman Problem (TSP) optimization** technique

### **Automatic Reverse Detection**
- Computes temporal feature correlation  
- If the sequence trend is reversed, the entire frame order is flipped automatically

### **Final Reconstruction**
- Sequentially writes all frames into a new MP4 file using **OpenCV VideoWriter**  
- Produces `reconstructed_video.mp4` as the final output  



##  **Design Considerations**

| Aspect | Design Choice |
|--------|----------------|
| **Accuracy** | Combination of deep CNN features + SSIM |
| **Speed** | Parallel computation using multithreading |
| **Scalability** | Efficient even for 300+ frames |
| **Robustness** | Automatically detects reversed order |
| **Ease of Use** | No training required — works on any single-shot video |



##  **Execution Time Log**

Execution details are stored in `Execution_Time_Log.txt`, including:
- Total execution time
- Feature extraction time
- Matrix calculation time
- TSP solving time 
- Number of Frames Processed
- Number of Comparisons performed
- Frames per Second (FPS) rate
- Output resolution



##  **Repository Structure**

```
JumbledVideoReconstruction/
│
├── reconstruct.py                # Main Python script
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
├── execution_log.txt        # Processing time log
├── reconstructed_video.mp4       # Final output video
└── docs/
    └── Algorithm-Explanation.md  # Detailed algorithm description
```



##  **Tech Stack and Tools**

| Tool / Library | Purpose |
|----------------|----------|
| **TensorFlow + Keras (ResNet50)** | Extracts deep semantic features |
| **OpenCV** | Frame extraction, video writing |
| **Scikit-image (SSIM)** | Computes pixel-level similarity |
| **NumPy, SciPy** | Numerical operations |
| **ThreadPoolExecutor** | Parallel computation |
| **Python 3.8+** | Main programming environment |



##  **Documentation**

Detailed explanation of the reconstruction algorithm and its optimization steps can be found in:  
📄 [`docs/Algorithm-Explanation.md`](docs/Algorithm-Explanation.md)






##  **Deliverables**

1. **Reconstructed Video** → `reconstructed_video.mp4`  
2. **Source Code** → `reconstruct.py`, `requirements.txt`  
3. **Algorithm Explanation** → `docs/Algorithm-Explanation.md`  
4. **Execution Time Log** → `execution_log.txt`  
5. **Public Repository** → Uploaded on GitHub






##  **Author**

**Name:** Marmik Kaila 


**Project:** Jumbled Frame Reconstruction Chellange


**Objective:** Restore temporal order in shuffled single-shot videos using AI and computer vision.  


**GitHub:** [https://github.com/MarmikKaila](https://github.com/MarmikKaila)

