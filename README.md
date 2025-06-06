# YouTube sponsor progress bar checker

A python script to check the progress of progress bars in YouTube videos

## Why?

Good question... I blame Matt Parker from [Stand-up Maths](https://www.youtube.com/@standupmaths). He released a video where he looked at the progress of progress bars in various YouTube videos. However he did a lot manually. This is why this project exists. My program isn't necessarily better, but it is pretty much hands free

Here is an example for the first sponsor segment in [Matt's video](https://www.youtube.com/watch?v=uc0OU1yJD-c)!

![A graph showing the progression of a progress bar](example_graph.png)

## How to run the script

First make sure you have the [SponsorBlock database](https://github.com/mchangrh/sb-mirror)

Then either move the `sponsorTimes.csv` to the base directory or edit your `config.json` and set `sponsorblockDatabasePath` to the full path of the `sponsorTimes.csv` file

### Edit your `config.json`

Set `searchType` to either 0, 1 or 2

0 will take the first few videos from the channel you put in `channelID`, `channelURLs` or `channelUsernames`

1 will use the videoIDs or videoURLs you put in `videoIDs` and `videoURLs`

2 will search YouTube for whatever you put in `search`

At this point you should be ready to go

> [!WARNING]
> I have only tested this on Linux! If you are on macOS you should be fine. But if you are on Windows, you may want to edit `getVideoid` and `getHead` in `commandlineTools` in your `config.json`

### Requirements

To install all the requirements, simply run:

```
pip install -r requirements.txt
```

### Running

To run the script:

```
python main.py
```

### Using the script with command line arguments

Currently there are only 2 explicitly defined command line arguments. Those are:

`-c` or `--config` to specify where your `config.json` file is located (e.g.: `python main.py -c ./config.json`)

and `-e` or `--export` to export the config file, after all the other command line arguments are applied (e.g.: `python main.py --export=true`)

Anything else you specify, will be used as a key to search through the config file (currently you have to use this option with `--`)

There are a few rules though:
- When using `-`, the value is the next argument
- When using `--`, the value is appended to the argument, only separated by an `=`
- Boolean values can be: t/f, true/false, y/n, yes/no. Of course in any case
- Lists and dicts are written exactly how you would in python, just there is no need for `"` or `'`
- Command line arguments can be given in: camelCase, PascalCase, snake_case, kebab-case. Don't mix and match though!
- Indexing into a list or dict is done by appending the index to the argument
- If a numerical index at the end of the argument is too large, the value is appended to the list
- You can use some predefined abbreviations
- abbreviations can also be used in the keys of dicts

Here are some examples:

`--SaveBaseDirectory=./`, `--saveBaseDirectory=./`, `--save-base-dir=./` and `--save_base_dir=./` do exactly the same thing: they sets `saveBaseDirectory` to `./` in `config.json`

`--search0ChannelIds="[a,b,1]"` sets `channelIds` to `['a', 'b', 1]` of the first item in the list that is `search` in `config.json`

`--search1="{melm:blep}"` will either append `{'melm': 'blep'}` to or replace the second item of `search` in `config.json`

`--startOfBarMax%=0.98` sets `startOfBarMaximumPercentage` to `0.98` in `config.json`

All the available abbreviations:

<pre>
    dir  ->  directory
    db  ->  database
    res  ->  resolution
    prog  ->  progressive
    sel  ->  selection
    func  ->  function
    idx  ->  index
    >  ->  direction
    %  ->  percentage
    thre  ->  threshold
    class  ->  classifier
    comp  ->  comparison
    min  ->  minimum
    max  ->  maximum
    diff  ->  difference
    pos  ->  positions
    gen  ->  generate
    hl  ->  highlight
    def  ->  definition
    ext  ->  extension
    color  ->  colour
    sb  ->  sponsorblock
</pre>

> [!WARNING]
> I have not extensively tested this! But it did work with most of the things I thew at it...



That's all! Have fun!
