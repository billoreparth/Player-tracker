# Football Player and Ball Tracker 

## 📑 Index
- [Overview](#football-player-and-ball-tracker)
- [Run Locally](#run-locally)

---


# Football Player and Ball Tracker 

This project uses a fine-tuned Torch model for real-time detection and tracking of football players, referees, goalkeepers, and the ball in match footage. Each player is assigned a unique ID and tracked throughout the video using ByteTrack, ensuring consistent identification even when players leave and re-enter the frame. The ball's position is tracked separately to analyze game dynamics. The system enables detailed analysis by distinguishing between referees, goalkeepers, and outfield players.


## Run Locally

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


