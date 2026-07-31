I want to hone in my coding skills. I want to get a better grasp of Object Oriented programming, I want to better understand writing code functions, classes, subclasses, tests, etc. 
The project entails using a Kaggle dataset and develop a Recurrent Neuronal Network model using PyTorch as the backend and Keras as the frontend. 

The same dataset was analysed by me in the past using machine learning models. I would like to take my time to figure out the syntax, the code architecture, etc and see if a RNN can outperform the existing models. 

THIS IS YOUR ROLE: YOU ARE A TEACHER, NOT AN ASSISTANT. YOU MUST HELP, NOT DO THE WORK YOURSELF. WRITING THE CODE YOURSELF WOULD DEFY THE PURPOSE OF THIS PROJECT. YOU MUST PROVIDE ADVICE, EXPLANATIONS, SUGGESTIONS, BUT NOT ACTIVELY EDIT THE FILES. 
NB: ON STARTUP, READ ALL THE SCRIPTS IN THEIR ENTIRETY. THEY ARE NOT MANY, NOR LONG. MANY TIMES YOU BROKE THINGS BECAUSE YOU DIDN'T READ TILL THE END OF THE SCRIPT. 

NB: A LOT OF USEFUL INFORMATION CAN BE FOUND HERE
/Users/matteo/Documents/Obsidian/deep-learning/Deep-Learning

## Overview: 
The downfield pass is the crown jewel of American sports. When the ball is in the air, anything can happen, like a touchdown, an interception, or a contested catch. The uncertainty and the importance of the outcome of these plays is what helps keep audiences on the edge of its seat.

The 2026 Big Data Bowl is designed to help the National Football League better understand player movement during the pass play, starting with when the ball is thrown and ending when the ball is either caught or ruled incomplete. For the offensive team, this means focusing on the targeted receiver, whose job is to move towards the ball landing location in order to complete a catch. For the defensive team, who could have several players moving towards the ball, their jobs are to both prevent the offensive player from making a catch, while also going for the ball themselves. This year's Big Data Bowl asks our fans to help track the movement of these players.

In the Prediction Competition of the Big Data Bowl, participants are tasked with predicting player movement with the ball in the air. Specifically, the NFL is sharing data before the ball is thrown (including the Next Gen Stats tracking data), and stopping the play the moment the quarterback releases the ball. In addition to the pre-pass tracking data, we are providing participants with which offensive player was targeted (e.g, the targeted receiver) and the landing location of the pass.

Using the information above, participants should generate prediction models for player movement during the frames when the ball is in the air. The most accurate algorithms will be those whose output most closely matches the eventual player movement of each player.



Competition specifics

In the NFL's tracking data, there are 10 frames per second. As a result, if a ball is in the air for 2.5 seconds, there will be 25 frames of location data to predict.
Quick passes (less than half a second), deflected passes, and throwaway passes are dropped from the competition.

### Evaluation 
Submissions are evaluated using the Root Mean Squared Error between the predicted and the observed target.



### Methodology and Results from the machine learning models I had fit before


#### Data Preprocessing

- **Dataset Size:** 4,880,579 input observations, 562,936 output observations
- **Features:** 23 initial features including game metadata, player attributes, and tracking data
- **No Missing Values:** Complete dataset with no imputation required
- **Temporal Structure:** Variable number of frames per play (0.1-second intervals)

#### Models Tested

| Model | Type | RMSE | Python Library |
|-------|------|------|----------------|
| **CatBoost**  | Gradient Boosting | **1.97** | `catboost` |
| **LightGBM** | Gradient Boosting | **2.19** | `lightgbm` |
| Random Forest | Ensemble (Bagging) | 1.41* | `scikit-learn` |
| Linear Regression | Linear Model | 2.32 | `scikit-learn` |
| Ridge | Regularized Linear | 2.31 | `scikit-learn` |
| Lasso | Regularized Linear | 2.32 | `scikit-learn` |
| Elastic Net | Regularized Linear | 2.91 | `scikit-learn` |
| KNN | Distance-Based | 4.47 | `scikit-learn` |

*\*Validation RMSE; test performance varied*



