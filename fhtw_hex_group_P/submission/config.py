BOARD_SIZE = 5
LEARNING_RATE = 1e-3
EPOCHS = 5000
GAMES_PER_EPOCH = 50
BATCH_SIZE = 64
patience = 100  # epochs to wait for improvement before stopping
no_improve_epochs = 0

new_games_played_in_epoch = 100  #number of epochs until the model plays new games
step_penalty = -0.01  # penalty for each step taken