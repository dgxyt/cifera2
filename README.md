# Cifera2
A short machine learning project and CLI utility using Tensorflow to generate models and predict the genres of music files.

Version: 2.0.0

Note: Whenever a relative file path is inputted, please use .\ to ensure compatibility.

Commands:
    cat:
        Concatenates two .csv files together, so long as they have the same header. The header will remain on the first line and will not be repeated. The final file will be outputted in the Cifera2 directory.

        ARGUMENTS:
            paths: The paths to each csv file. An unlimited number of paths may be used.
            
    generate:
        Generates a model using the online dataset, optionally in addition to a custom dataset.

        ARGUMENTS:
            None

        OPTIONS:
            --data: The path to a .csv file containing additional training data.
            --exclude-builtin-dataset: exclude the default dataset that is used.
            --name: The custom name of the model to be created.

    format:
        Formats the first 30 seconds of an audio file to be evaluated using a Cifera2 model.

        ARGUMENTS:
            path: The path to the unformatted audio file.

        OPTIONS:
            --genre: The genre of the audio file. Note: Specifying genre highly recommended when using the resulting formatted files to train. Default: "unknown"

    batchformat:
        Formats the first 30 seconds of each audio file in a directory of audio files to be evaluated using a Cifera2 model.

        ARGUMENTS:
            path: The path to the folder of unformatted audio files. The directory must exclusively contain audio files.

        OPTIONS:
            --genre: The genre of all audio files in the folder. Note: Specifying genre highly recommended when using the resulting formatted files to train. Default: "unknown"
            --seperate: Whether or not to output each formatted file into an individual .csv file, or whether to group them all into a singular .csv. Default: Off
        
    evaluate:
        Evaluates a formatted audio file using a Cifera2 model.

        ARGUMENTS:
            path: The path to the formatted audio file to be evaluated

        OPTIONS:
            --graph: Display a graph of the finished confidence chart. Default: Off
    
    purge:
        Removes all model and artifact files.

        ARGUMENTS:
            None

        OPTIONS:
            None

Potential predicted genres:
    "Blues"
    "Classical"
    "Country"
    "Disco"
    "Hiphop"
    "Jazz"
    "Metal"
    "Pop"
    "Reggae"
    "Rock"

Config: config.json
    commands: Configuration for each command.
        evaluate: Configuration for the evaluate command
            model_version: Which version of the Cifera2 model to use for evaluation. See models folder for models. Default: empty (A model must first be generated.) Example: "2"
    model: Configuration of the model and its layers.
        layers: An array containing, in order, each of the layers of the Sequential model. Default: ["Dense(256, activation='relu', input_shape=(X_train.shape[1],))","BatchNormalization()","Dropout(0.4)","Dense(128, activation='relu')","BatchNormalization()","Dropout(0.3)","Dense(64, activation='relu')","Dropout(0.2)","Dense(y_onehot.shape[1], activation='softmax')"],
        optimizer: The optimizer to use when training. Default: "adam"
        loss: The loss algorithm to use when training. Default: "categorical_crossentropy"
        epochs: The maximum amount of epochs to train for. Default: 100
        batch_size: Batch size while fitting. Default: 32
        name: The default version name of the model. Default: "2"

Other files:
    Cipher\Cipher.exe
        Use the Cipher utility to fetch and trim YouTube links, yielding 30 second .wav files that are ready to be formatted by Cifera2.
        Input a list of requested videos using Cipher\input.txt, with each line following the format [link],[genre]. After running Cipher, each .wav file will be saved in the Cipher\datasets folder.
        Note: ffmpeg is required on the system for the Cipher utility to function. See https://ffmpeg.org/download.html to download ffmpeg for your system.

Included Datasets:
    "Music features" by Insiyah Hajoori and MARSYAS.
