# Cifera2

A short machine learning project and CLI utility using TensorFlow to generate models and predict the genres of music files.

**Version:** 2.0.0  
**Note:** When providing relative file paths, please use `.\` for compatibility.  
**Note:** To run the CLI program, use the command `python c2.py` (python version) or `c2` (executable version).

---

## Commands

### `cat`
Concatenates two or more `.csv` files, assuming they share the same header. The header will remain only on the first line. The output will be saved in the Cifera2 directory.

**Arguments:**
- `paths`: The paths to each `.csv` file. You may use an unlimited number of paths.

---

### `generate`
Generates a model using the online dataset, optionally including a custom dataset.

**Arguments:**
- None

**Options:**
- `--data`: Path to a `.csv` file with additional training data.
- `--exclude-builtin-dataset`: Excludes the default dataset.
- `--name`: Custom name for the model to be created.

---

### `format`
Formats the first 30 seconds of a single audio file for evaluation by a Cifera2 model.

**Arguments:**
- `path`: Path to the unformatted audio file.

**Options:**
- `--genre`: The genre of the audio file.  
  **Recommended** when using formatted files for training.  
  *Default:* `"unknown"`

---

### `batchformat`
Formats the first 30 seconds of **each** audio file in a directory for evaluation.

**Arguments:**
- `path`: Path to a folder containing **only** audio files.

**Options:**
- `--genre`: Genre applied to all audio files.  
  *Default:* `"unknown"`
- `--seperate`: Whether to output each formatted file as an individual `.csv` or combine all into one.  
  *Default:* Off

---

### `evaluate`
Evaluates a **formatted** audio file using a Cifera2 model.

**Arguments:**
- `path`: Path to the formatted `.csv` file.

**Options:**
- `--graph`: Displays a graph of the prediction confidence.  
  *Default:* Off

---

### `purge`
Deletes all model and artifact files.

**Arguments:** None  
**Options:** None

---

## Potential Predicted Genres

- Blues
- Classical
- Country
- Disco
- Hiphop
- Jazz
- Metal
- Pop
- Reggae
- Rock

---

## Configuration

### `config.json`

#### `commands`
- **`evaluate`**
  - `model_version`: Specifies which version of the model to use for evaluation.  
    Example: `"2"`  
    *Default:* empty (you must first generate a model)

#### `model`
- **`layers`**  
  *Default:*
  ```python
  [
    "Dense(256, activation='relu', input_shape=(X_train.shape[1],))",
    "BatchNormalization()",
    "Dropout(0.4)",
    "Dense(128, activation='relu')",
    "BatchNormalization()",
    "Dropout(0.3)",
    "Dense(64, activation='relu')",
    "Dropout(0.2)",
    "Dense(y_onehot.shape[1], activation='softmax')"
  ]
- **`optimizer`**  
  *Default:* `"adam"`

- **`loss`**  
  *Default:* `"categorical_crossentropy"`

- **`epochs`**  
  *Default:* `100`

- **`batch_size`**  
  *Default:* `32`

- **`name`**  
  *Default:* `"2"`

## Included Datasets
"Music features" by Insiyah Hajoori and MARSYAS
