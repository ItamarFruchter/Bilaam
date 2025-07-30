# Bilaam
Biblical Large Actully Acronymless Model (or BILAAM) is my attempt at building an LLM from (almost) scratch. Following the wonderfull [Andrej Karpathy](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) tutorial I wanted to create something of my own (the original task - shakespear based model).

And sadly, for this project and my learning, copilot and cursor were turned off for this whole project (including the readme!)

This model is trained on the whole hebrew bible, and can generate brand new verses on whatever topic you'd like!

## Tech
### Tokenization
Is based on BLE tokenization with a bit of regex splitting (to remove some undesired unifications). This probably isn't optimal for hebrew, and the fact that the model is generating byte level outputs and the output is then decoded with UTF-8 it generates some illegible tokens. Forgive it!

### Model
Is composed of embedding for both position and tokens, a number of attention blocks, and finaly a linear layer.
The attention here is soley self-attention.
Because it hasn't been through the process of fine-tuning to answer user inputs, it won't reply to your messages, but depending on your belief so does the bible 😉

## Usage
To play with the model, clone the project and run the BilaamPlayground.ipynb notebook!
You can train your own version with the code there as well, and play with the parameters to try and get a better model.
This one was trained localy on a modest GPU.

## Examples
>איתמר בחור יעל-אשת חרפה בנחל, פליא-חרב.

>שם טוב לכלב: גם-אבדאל, בא בילי;

>בילעם הוא מודל שפה כי קטב, ואין אין לב
