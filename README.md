# Classification of Affective Emotion in Soundtracks

### How to understand the emotinal content of the soundtracks of the movies?

## Definition

Music Emotion Recognition (MER) is a challenging problem due to:

1. Subjectiveness of emotions
2. Lack of suitable datasets
3. Need of complex algorithms to analyse sequential data


## Goal

Train a model to predict the emotional content of soundtracks. For that we tested two types of audio features (MFCCs and VGGish) on two networks (CNN and LSTM) to see which one performed the best.

## Dataset

There are two main approaches to represent emotions:

1. Categorical: mood labels. Ex. sadness, disgust, anger, fear, surprise
2. Dimensional: numerical values

A numerical representation of emotions was needed for this type of work, so we chose Russell's dimensional model, which reprensent emotions based on three values:

- **Valence**: the positive or negative
- **Arousal**: the exciment or relaxation
- **Dominance**: the level of control or submission


### How the dataset was created?

1. Soundtracks were retrieved from [lastfm api](https://www.last.fm/api) obtaining a total of 9000 records with the *soundtrack* tag. Each song has been labeled by the users from last.fm platform with keywords and their associated weights. Here is an output example:

```python
{'The End Of The World', 'Skeeter Davis',
    {'sad': 10,
    'lost': 10,
    'romantic': 6,
    'love': 5,
    'nice': 4,
    'beautiful': 3,
    'suicide': 3,
    'rock': 2,
    'power': 2,
    'slow': 2,
    'peace': 2,
    'smooth': 2,
    'soft': 2,
    'fun': 2}}
```

2. To convert those keywords into emotions measured with numerical values (vad), the (NRC dataset)[https://www.kaggle.com/datasets/wjburns/nrc-emotion-lexicon/data] was employed. This dataset has 20000 english words with the valence, arousal and dominance associated with it. Ex:

| Word    | Valence | Arousal | Dominance |
|---------|---------|---------|-----------|
| Sad     | 0.225   | 0.333   | 0.149     |
| Happy   | 1       | 0.735   | 0.772     |
| Fear    | 0.073   | 0.84    | 0.293     |

3. For each song we calculated the total vad based on the weighted average of the vad of each keyword. Ex:

**Simple as This – Jake Bugg**

| Mood       | Weight | Valence | Arousal | Dominance |
|------------|--------|---------|---------|-----------|
| Cool       | 56     | 0.885   | 0.930   | 0.653     |
| Nice       | 12     | 0.540   | 0.442   | 0.361     |
| Study      | 12     | 0.781   | 0.650   | 0.768     |
| **Total**  | 80     | 0.856   | 0.498   | 0.759     |

- **Valence**: (0.885 * 56 + 0.540 * 12 + 0.781 * 12) / 80 = **0.856**
- **Arousal**: (0.930 * 56 + 0.442 * 12 + 0.650 * 12) / 80 = **0.498**
- **Dominance**: (0.653 * 56 + 0.361 * 12 + 0.768 * 12) / 80 = **0.759**

4. From last.fm we took the spotify id to get the actual audio file with the `savify` library. The actual audio is needed to extract the MFCC and VGGish features.


## Conclusion

We trained a CNN and a LSTM with this data in order to predict the emotions on different soundtracks and we arrived to these conclusions:

- MFCCs are more pre-processed so they lose their temporal relationship. They perform better on CNN as they resemble more to an image.

- VGGish are less pre-processed so they have a stronger temporal audio relationship and so they perform better on LSTM. This configuration gave the best results.

Results on **Valence**

| Network | Feature |  MSE   |  MAE   |
|---------|---------|--------|--------|
| CNN     | MFCC    | 0.0153 | 0.0977 |
| LSTM    | MFCC    | 0.0153 | 0.0966 |
| CNN     | VGGish  | 0.0143 | **0.0919** |
| **LSTM**    | **VGGish**  | **0.0141** | 0.0919 |

Results on **Arousal**

| Network | Feature |  MSE    | MAE    |
|---------|---------|---------|--------|
| CNN     | MFCC    |  0.0095 | 0.0761 |
| LSTM    | MFCC    |  0.0098 | 0.0773 |
| CNN     | VGGish  |  0.0091 | 0.0738 |
| **LSTM**    | **VGGish**  |  **0.0089** | **0.0733** |


Results on **Dominance**

| Network | Feature |  MSE    | MAE    |
|---------|---------|---------|--------|
| CNN     | MFCC    |  0.0084 | 0.0694 |
| LSTM    | MFCC    |  0.0084 | 0.0773 |
| CNN     | VGGish  |  0.0079 | 0.0669 |
| **LSTM**    | **VGGish**  |  **0.0077** | **0.0662** |