
## Index
- [Overview](#football-player-and-ball-tracker)
- [Installation](#installation)
- [Run From New Stubs](#run-from-new-stubs)
- [Screenshots](#screenshot)
- [Demo](#demo-output)

---


# Football Player and Ball Tracker 

This project uses a fine-tuned Torch model for real-time detection and tracking of football players, referees, goalkeepers, and the ball in match footage. Each player is assigned a unique ID and tracked throughout the video using ByteTrack, ensuring consistent identification even when players leave and re-enter the frame. The ball's position is tracked separately to analyze game dynamics. The system enables detailed analysis by distinguishing between referees, goalkeepers, and outfield players.

---


# Installation

Step 1: Clone the project

```bash
  git clone https://github.com/billoreparth/Player-tracker.git
```

Step 2: Download the Torch Model 'best.pt' in project directory
```
https://drive.google.com/file/d/1H2NnVn9o4fucVP2AaSmuapztHqygmNIN/view?usp=sharing
```
(Optional): Make an python 3.10.18 environment and Activate it
```
python3.10.18 -m venv <environment-name>
```
``` 
<environment-name>\Scripts\activate
```


Step 3: Go to the project directory and install dependencies 

```bash
  pip install -r requirements.txt 
```
Step 4: Run the main.py file 

```bash
  python -u <main.py file path>
```

Step 5: You will find the output video in project directory
```
<project directory>\output-videos\
 ```
---
# Run From New Stubs
A stub in this project is a precomputed file that stores detection and tracking results so the program doesn't have to re-run the entire model again every time \
To change that and Re-run the model without loading the previous stubs \
In 'main.py' file in line no. 9 , in 'get_objects_tracks' function change the 'read_from_stubs' to False and Do Not provide any 'stub_path'
```
read_from_stub=False
```
---

# Screenshot

Sample :

<img width="1919" height="1079" alt="Screenshot 2025-07-13 121527" src="https://github.com/user-attachments/assets/31d49655-4a94-4958-957a-2916de570602" />


Predicted by Model :

<img width="1919" height="1079" alt="Screenshot 2025-07-13 121552" src="https://github.com/user-attachments/assets/6571f55c-84fb-4878-8c3f-df2ab346d23d" />


Tracked Output :

<img width="1919" height="1079" alt="Screenshot 2025-07-13 121459" src="https://github.com/user-attachments/assets/9dd916b5-b21f-492d-8e0f-fcfa1c2ce5f4" />

---
# Demo Output  


https://github.com/user-attachments/assets/05179544-4644-4347-b0e6-cce071eccea8


---
