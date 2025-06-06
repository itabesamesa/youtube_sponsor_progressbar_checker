import scrapetube
from pytubefix import YouTube, Stream
from pytubefix.cli import on_progress
from pytubefix.contrib.search import Search, Filter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import ffmpeg
import json
import random
import re
import platform
import sys

configPath = "config.json"

def to_camel_case(ogString):
    string = ogString[0].lower()+ogString[1:]
    string = string.split("-")
    from_snake_or_kebab = lambda x: x[0]+"".join([x[i].capitalize() for i in range(1, len(x))])
    if (len(string) > 1):
        return [from_snake_or_kebab(string), ogString.split("-")]
    else:
        string = string[0].split("_")
        if (len(string) > 1):
            return [from_snake_or_kebab(string), ogString.split("_")]
    return [string[0], re.split("(?<=[a-zA-Z])(?=[A-Z])|(?<=[0-9])(?=[A-Z])|(?<=[a-zA-Z])(?=[0-9])", ogString)]

abbreviations = {
    ("dir", "directory"),
    ("db", "database"),
    ("res", "resolution"),
    ("prog", "progressive"),
    ("sel", "selection"),
    ("func", "function"),
    ("idx", "index"),
    (">", "direction"),
    ("%", "percentage"),
    ("thre", "threshold"),
    ("class", "classifier"),
    ("comp", "comparison"),
    ("min", "minimum"),
    ("max", "maximum"),
    ("diff", "difference"),
    ("pos", "positions"),
    ("gen", "generate"),
    ("hl", "highlight"),
    ("def", "definition"),
    ("ext", "extension"),
    ("color", "colour"),
    ("sb", "sponsorblock")
}

def expand_abbreviation(string):
    to_regex = lambda x: "^"+x+"(?=$|[^a-z])|(?<=.)"+x.capitalize()+"(?=$|[^a-z])"
    #to_test = lambda x: x+"Stuff"+x.capitalize()+"Stuff"+x+"Stuff"+x.capitalize()+"stuff"+x.capitalize()
    for i in abbreviations:
        string = i[1].capitalize().join(re.split(to_regex(i[0]), string))
        string = string[0].lower()+string[1:]
    return string

def get_value_type(string):
    match string:
        case str():
            if (re.fullmatch("[0-9]+", string)):
                return int(string)
            elif (re.fullmatch("[0-9]*\\.[0-9]+", string)):
                return float(string)
            elif (re.fullmatch("t|true|y|yes", string.lower())):
                return True
            elif (re.fullmatch("f|false|n|no", string.lower())):
                return False
            elif (re.fullmatch("^\\[.*\\]$|^\\{.*\\}$", string)):
                string = get_value_type(eval(re.sub("(?<=[\\[\\{\\:,])(?=[^\\[\\{\\:,])|(?<=[^\\]\\}\\:,])(?=[\\]\\}\\:,])", "\"", string)))
                return string
            else:
                return string
        case list():
            return [get_value_type(x) for x in string]
        case dict():
            return {expand_abbreviation(k): get_value_type(v) for k, v in string.items()}
        case _:
            return string

argValuePairs = []
exportConfig = False
i = 1
while i < len(sys.argv):
    baseArg = sys.argv[i].split("=")
    if (len(baseArg) > 1):
        arg = baseArg.pop(0)[2:]
        value = "=".join(baseArg)
        complexArg = True
    else:
        arg = baseArg[0][1:]
        i += 1
        value = sys.argv[i]
        complexArg = False
    print("arg:       '"+arg+"'")
    print("value:     '"+value+"'")
    match arg: #match incase i ever want to explicitly define more options
        case "c" | "config":
            if (not(os.path.isfile(value))):
                sys.exit("File '"+value+"' doesn't exist. Please choose a valid 'config.json' file")
            configPath = value
        case "e" | "export":
            value = get_value_type(value)
            if not type(value) is bool:
                sys.exit("Value for '"+arg+"' must be a boolean")
            exportConfig = value
        case _:
            cameled = to_camel_case(arg)
            expArg = expand_abbreviation(cameled[0])
            value = get_value_type(value)
            argValuePairs.append([expArg, value, complexArg, arg, cameled[1]])
    i += 1

print()
print(configPath)
print(argValuePairs)
print()

with open(configPath, 'r') as file:
    config = json.load(file)

def search_config_complex(opt, arg, available, config=config, search=""):
    if (len(arg) == 0): return True
    search += arg.pop(0)
    opt[4].pop(0)
    if search in available:
        print("found match")
        if (len(arg) == 0):
            print("Valid argument: '--"+opt[3]+"'")
            print("Setting in config")
            print("  "+str(config[search]))
            config[search] = opt[1]
            print("  "+str(config[search]))
            return False
        else:
            opt2 = opt.copy()
            opt2[4] = opt[4].copy()
            result = search_config_complex(opt2, arg.copy(), available, config, search)
            del opt2[4]
            del opt2
            if (result):
                config = config[search]
                index = arg.pop(0)
                if (len(arg) == 0):
                    if type(config) is list:
                        if (re.fullmatch("[0-9]+", index)):
                            index = int(index)
                        else:
                            sys.exit("Invalid argument: '--"+opt[3]+"'. Index '"+opt[4][0]+"' is not a number")
                            return False
                        if(index >= len(config)):
                            print("Valid argument: '--"+opt[3]+"'")
                            print("Setting in config")
                            print("Appending to list")
                            config.append(opt[1])
                            print("  "+str(config))
                            return False
                        else:
                            print("Valid argument: '--"+opt[3]+"'")
                            print("Setting in config")
                            config[index] = opt[1]
                            print("  "+str(config[index]))
                            return False
                    elif type(config) is dict:
                        if index in config.keys():
                            print("Valid argument: '--"+opt[3]+"'")
                            print("Setting in config")
                            config[index] = opt[1]
                            print("  "+str(config[index]))
                            return False
                    sys.exit("Invalid argument: '--"+opt[3]+"'. Unable to index")
                else:
                    print(type(index))
                    if type(config) is list:
                        if (re.fullmatch("[0-9]+", index)):
                            index = int(index)
                        else:
                            sys.exit("Invalid argument: '--"+opt[3]+"'. Index '"+opt[4][0]+"' is not a number")
                        if(index >= len(config)):
                            sys.exit("Invalid argument: '--"+opt[3]+"'. Index '"+opt[4][0]+"' is out of range")
                    elif type(config) is dict:
                        if not(index in config.keys()):
                            sys.exit("Invalid argument: '--"+opt[3]+"'. Unable to index")
                    else:
                        sys.exit("Invalid argument: '--"+opt[3]+"'. Unable to index")
                    config = config[index]
                    if not type(config) is dict:
                        sys.exit("Invalid argument: '--"+opt[3]+"'. Too many indicies or nested list")
                    arg[0] = arg[0].lower()
                    available = [x for x in config.keys() if (re.match("^"+arg[0], x))]
                    if (len(available) == 0):
                        sys.exit("Invalid argument: '--"+opt[3]+"'. No more options available")
                    opt[4].pop(0)
                    return search_config_complex(opt, arg, available, config)
            return False
    return search_config_complex(opt, arg, available, config, search)

for opt in argValuePairs:
    if (opt[2]):
        arg = re.split("(?<=[a-zA-Z])(?=[A-Z])|(?<=[0-9])(?=[A-Z])|(?<=[a-zA-Z])(?=[0-9])", opt[0])
        available = [x for x in config.keys() if (re.match("^"+arg[0], x))]
        if (len(available) == 0):
            sys.exit("Invalid argument: '--"+opt[0]+"'")
        elif (len(available) == 1):
            if (available[0] == opt[0]):
                print("Valid argument: '--"+opt[0]+"'")
                print("Setting in config")
                print("  "+str(config[opt[0]]))
                config[opt[0]] = opt[1]
                print("  "+str(config[opt[0]]))
            else:
                print("Invalid argument: '--"+opt[0]+"'")
        else:
            if any(opt[0] == x for x in available):
                print("Valid argument: '--"+opt[0]+"'")
                print("Setting in config")
                print("  "+str(config[opt[0]]))
                config[opt[0]] = opt[1]
                print("  "+str(config[opt[0]]))
            else:
                search_config_complex(opt, arg, available)
    else:
        print("Can't handle those yet. Sooo... I'm gonna skip")

if (config["saveBaseDirectory"][-1] != "/"):
    config["saveBaseDirectory"] += "/"

if (exportConfig):
    i = 0
    configPath = configPath.split(".")
    extension = configPath.pop()
    configPath = ".".join(configPath)
    while True:
        newConfigPath = configPath+str(i)+"."+extension
        if (os.path.isfile(newConfigPath)):
            i += 1
        else:
            print("Exported config to: '"+newConfigPath+"'")
            info = json.dumps(config, indent=2)
            with open(newConfigPath, "w") as outfile:
                outfile.write(info)
            break

del configPath

try:
    os.mkdir(config["saveBaseDirectory"])
except:
    pass

if (not(os.path.isfile(config["sponsorblockDatabasePath"]))):
    print("'"+config["sponsorblockDatabasePath"]+"' doesn't exist")
    file = os.path.basename(config["sponsorblockDatabasePath"])
    dir = config["sponsorblockDatabasePath"][:len(config["sponsorblockDatabasePath"])-len(file)]
    print("Either move '"+file+"' to '"+dir+"'")
    print("Or change 'sponsorblockDatabasePath' to where your SponsorBlock database is")
    print("If you haven't got a SponsorBlock database, download it from: https://github.com/mchangrh/sb-mirror")
    exit()

proxies = config["proxies"]
if (proxies["http"] == None and proxies["https"] == None):
    proxies = None
else:
    proxies = {k: v for k, v in proxies.items() if v is not None}

operatingSystem = platform.system()
if (operatingSystem != "Linux"):
    print("WARNING!")
    print("This was only tested on Linux")
    print("You are on "+operatingSystem)
    if (os == "Windows"):
        if (re.search("grep", config["commandlineTools"]["getVideoId"])):
            print("  You are using 'grep' in 'commandlineTools' 'getVideoId'")
            print("  Please change your config")
            print("  For now setting to: 'findstr /c:\"$id\"'")
            config["commandlineTools"]["getVideoId"] = "findstr /c:\"$id\""
        if (re.search("head", config["commandlineTools"]["getHead"])):
            print("  You are using 'head' in 'commandlineTools' 'getHead'")
            print("  Please change your config")
            print("  For now setting to: 'findstr /b videoID'")
            config["commandlineTools"]["getHead"] = "findstr /b videoID"
    elif (operatingSystem == "Java"):
        print("  Idk if you'll be fine... Yolo?")
    elif (operatingSystem in ["Darwin", "iOS", "iPadOS", "Android"]):
        print("  You should be fine? Fingers crossed...")
    else:
        print("  This shouldn't have happend...")
        print("  Continue with caution")
    answer = input("Are you sure you want to continue? [y/N] ")
    if not(answer.lower() in ["y", "yes"]):
        exit()

grep = re.split("[$][$]", config["commandlineTools"]["getVideoId"])
if (not(any(re.search("[$]id", x) for x in grep))):
    grep[-1] += " \""+config["commandlineTools"]["runTimeId"]+"\""
if (not(any(re.search("[$]path", x) for x in grep))):
    grep[-1] += " "+config["sponsorblockDatabasePath"]
grep = "$".join([re.sub("[$]path", config["sponsorblockDatabasePath"], re.sub("[$]id", config["commandlineTools"]["runTimeId"], x)) for x in grep])
config["commandlineTools"]["getVideoId"] = grep
del grep

head = re.split("[$][$]", config["commandlineTools"]["getHead"])
if (not(any(re.search("[$]path", x) for x in head))):
    head[-1] += " "+config["sponsorblockDatabasePath"]
head = "$".join([re.sub("[$]path", config["sponsorblockDatabasePath"], x) for x in head])
config["commandlineTools"]["getHead"] = head
del head

videoIDs = set()

match config["searchType"]:
    case 0:
        search = config["search"][config["searchType"]]
        print("Search with scrapetube by channelID")
        if (search["channelIds"] == [] and search["channelUrls"] == [] and search["channelUsernames"] == []):
            print("No values provided for 'channelIds', 'channelUrls' and 'channelUsernames'")
            exit()
        contentType = None
        if (search["contentType"] == None):
            contentType = "videos"
        elif (["videos", "shorts", "streams"].count(search["contentType"]) != 0):
            contentType = search["contentType"]
        else:
            print("Invalid type '"+str(search["searchType"])+"' for 'contentType'")
            exit()
        for i in search["channelIds"]:
            videos = scrapetube.get_channel(
                channel_id=i,
                limit=search["limit"],
                sleep=search["sleep"],
                #proxies=proxies, #TypeError: get_channel() got an unexpected keyword argument 'proxies'
                sort_by=search["sortBy"],
                content_type=contentType)
            for video in videos:
                videoIDs.add(video["videoId"])
        for i in search["channelUrls"]:
            videos = scrapetube.get_channel(
                channel_url=i,
                limit=search["limit"],
                sleep=search["sleep"],
                #proxies=proxies, #TypeError: get_channel() got an unexpected keyword argument 'proxies'
                sort_by=search["sortBy"],
                content_type=contentType)
            for video in videos:
                videoIDs.add(video["videoId"])
        for i in search["channelUsernames"]:
            videos = scrapetube.get_channel(
                channel_username=i,
                limit=search["limit"],
                sleep=search["sleep"],
                #proxies=proxies, #TypeError: get_channel() got an unexpected keyword argument 'proxies'
                sort_by=search["sortBy"],
                content_type=contentType)
            for video in videos:
                videoIDs.add(video["videoId"])
    case 1:
        search = config["search"][config["searchType"]]
        print("Search with pytubefix by videoIDs")
        videoIDs = set(search["videoIds"])
        [videoIDs.add(x[2:]) for i in search["videoUrls"] for x in re.findall("v=[\\w\\-]*", i)]
    case 2:
        search = config["search"][config["searchType"]]
        print("Search with pytubefix by search term")
        i = search["limit"]
        f = search["filters"]
        filters = {}
        if (f["uploadDate"] != None):
            filters.update({"upload_date": Filter.get_upload_date(f["uploadDate"])})
        if (f["type"] != None):
            filters.update({"type": Filter.get_type(f["type"])})
        if (f["duration"] != None):
            filters.update({"duration": Filter.get_duration(f["duration"])})
        if (f["features"] != []):
            filters.update({"features": [Filter.get_features(x) for x in f["features"]]})
        if (f["sortBy"] != None):
            filters.update({"sort_by": Filter.get_sort_by(f["sortBy"])})
        if (filters == {}):
            filters = None
        s = Search(search["search"], proxies=proxies, filters=filters)
        if (i <= 0):
            for video in s.videos:
                videoIDs.add(video.video_id)
        else:
            while True:
                for video in s.videos:
                    l = len(videoIDs)
                    videoIDs.add(video.video_id)
                    if (len(videoIDs) != l):
                        i -= 1
                        if (i <= 0):
                            break
                else:
                    s.get_next_results()
                    continue
                break
    case _:
        print("Invalid option for 'searchType': "+str(config["searchType"]))
        exit()

print("Video IDs found:", videoIDs)

urlBase = config["urlBase"]

def mean(a):
    return sum(a)/len(a)

def segment_list(df, value, dist=200):
    all = []
    for i in df.index:
        v = df.at[i, value].item()
        if (all == []):
            all.append([v])
        else:
            app = True
            for x in all:
                av = mean(x)
                if ((v > av-dist) and (v < av+dist)):
                    x.append(v)
                    app = False
                    break
            if (app):
                all.append([v])
    return all

def get_videoid_from_sponsorblock_database(videoID):
    grep = re.sub(config["commandlineTools"]["runTimeIdRegex"], videoID, config["commandlineTools"]["getVideoId"])
    out = os.popen(grep).read()

    if (out == ""):
        print("No sponsor segments")
        return []

    out = os.popen(config["commandlineTools"]["getHead"]).read()+out

    df = pd.read_table(io.StringIO(out), delimiter=',')

    print("Database for "+videoID+":")
    print(df)

    for i in config["filterSponsorblockDatabase"]:
        if ((i["comparison"] == "in") ^ isinstance(i["value"], list)):
            print("'comparison' is incompatible for type of 'value'. Skipping")
            print("  "+str(i))
            continue
        pyCmd = " "+i["comparison"]+" "+("\""+str(i["value"])+"\"" if (isinstance(i["value"], str)) else str(i["value"]))
        stringify = lambda series: ("\""+str(series.loc[i["column"]])+"\"") if (type(series.loc[i["column"]]) == str) else str(series.loc[i["column"]])
        df = df.loc[[eval(stringify(series)+pyCmd) for idx, series in df.iterrows()]]

    if df.empty:
        print("No sponsor segments")
        print("Try adjusting 'filterSponsorblockDatabase'")
        return []

    print("Filtered database for "+videoID+":")
    print(df)

    startTimes = segment_list(df, "startTime")
    endTimes = segment_list(df, "endTime")

    del df

    if (len(startTimes) != len(endTimes)):
        print("Start and end times are inconsistent. Skipping")
        return []

    startTimesAvg = [mean(i) for i in startTimes]
    endTimesAvg = [mean(i) for i in endTimes]
    startTimesAvg.sort()
    endTimesAvg.sort()

    sponsor = [{"start": i, "end": endTimesAvg[idx], "duration": endTimesAvg[idx]-i} for idx, i in enumerate(startTimesAvg)]
    return sponsor

videos = []
for i in videoIDs:
    video = YouTube(urlBase+i, on_progress_callback=on_progress)
    videos.append(
        {
            "video": video,
            "videoID": i,
            "videoURL": urlBase+i,
            "videoTitle" : video.title,
            "sponsor": get_videoid_from_sponsorblock_database(i),
            "duration": video.length
        }
    )

youTubeFilter = config["filterYouTubeStreams"]

for i in videos:
    i["video"] = i["video"].streams.filter(
        fps=youTubeFilter["fps"],
        resolution=youTubeFilter["resolution"],
        mime_type=youTubeFilter["mimeType"],
        type=youTubeFilter["type"],
        subtype=youTubeFilter["subtype"],
        bitrate=youTubeFilter["bitrate"],
        video_codec=youTubeFilter["videoCodec"],
        audio_codec=youTubeFilter["audioCodec"],
        progressive=youTubeFilter["progressive"],
        is_dash=youTubeFilter["isDash"],
        only_video=youTubeFilter["onlyVideo"],
        audio_track_name=youTubeFilter["audioTrackName"]
    ).order_by(config["orderBy"])

match config["selectionFunction"]:
    case "first":
        for i in videos:
            i["video"] = i["video"].first()
    case "last":
        for i in videos:
            i["video"] = i["video"].last()
    case "index":
        for i in videos:
            i["video"] = i["video"][config["index"] if (config["index"] < len(i)) else -1]
    case _:
        print("Invalid value '"+config["selectionFunction"]+"' for 'selectionFunction'")
        print("Continuing with 'first'")
        for i in videos:
            i["video"] = i["video"].first()

crop = config["crop"]

for video in videos:
    video.update({"size": {"width": video["video"].width, "height": video["video"].height}})

    print("Processing...")
    print("Video ID: '"+video["videoID"]+"'")
    print("Video title: '"+video["videoTitle"]+"'")
    if (video["sponsor"] == None and config["skipNotSponsored"]):
        print("No sponsor found! Skipping")
        continue
    path = config["saveBaseDirectory"]+video["videoID"]
    try:
        os.mkdir(path)
    except:
        if (config["ignoreExistingDirectory"]):
            print("Directory already exists! Skipping")
            continue

    videoPath = video["video"].download(output_path=path, skip_existing=not(config["alwaysDownload"]))
    video.update({"videoPath": videoPath, "directory": path})

    videoCrop = {
        "x": int(crop["x"]*video["video"].width),
        "y": int(crop["y"]*video["video"].height)
    }
    videoCrop.update({
        "width": int(crop["width"]*(video["video"].width-videoCrop["x"])),
        "height": int(crop["height"]*(video["video"].height-videoCrop["y"]))
    })
    video.update({"sponsorSize": {"width": videoCrop["width"], "height": videoCrop["height"]}})

    print()
    print("Found "+str(len(video["sponsor"]))+" sponsorships")
    print()

    for idx, i in enumerate(video["sponsor"]):
        sponsorPath = path+"/"+str(idx)+".mp4"
        sponsorPart = (ffmpeg
            .input(videoPath)
            .trim(start=i["start"], end=i["end"])
            .crop(
                x=videoCrop["x"],
                y=videoCrop["y"],
                width=videoCrop["width"],
                height=videoCrop["height"]
            )
        )
        if (config["overwriteExistingFiles"]):
            sponsorPart = ffmpeg.output(sponsorPart, sponsorPath, y=None)
        else:
            sponsorPart = ffmpeg.output(sponsorPart, sponsorPath, n=None)
        ffmpeg.run(sponsorPart, quiet=config["ffmpegQuiet"])
        i.update({"path": sponsorPath, "index": idx})

    if (config["deleteAfterFinish"]):
        os.remove(videoPath)

print("Finished getting and formating videos")

def guess_bar_var(avg):
    var = []
    for i in range(avg.shape[0]):
        sample = avg[max(0, i+config["barDirectionScanRange"][0]):min(i+config["barDirectionScanRange"][1], avg.shape[0]-1)]
        var.append(sample.var(axis=0))

    addTo = 0
    similar = [{"avg": var[0], "amount": 1, "start": 0}]
    for i in range(1, len(var)):
        a = np.array([similar[addTo]["avg"], var[i]])
        b = a.var(axis=0).max().item()
        if (b < config["barDirectionClassifierThreshold"]):
            similar[addTo]["avg"] = a.mean(axis=0)
            similar[addTo]["amount"] += 1
        else:
            similar.append({"avg": var[i], "amount": 1, "start": i})
            addTo += 1

    #return max(similar, key=lambda x: x["amount"])
    return similar

def guess_bar_std(avg):
    std = []
    for i in range(avg.shape[0]):
        sample = avg[max(0, i+config["barDirectionScanRange"][0]):min(i+config["barDirectionScanRange"][1], avg.shape[0]-1)]
        std.append(sample.std(axis=0))

    addTo = 0
    similar = [{"avg": std[0], "amount": 1, "start": 0}]
    for i in range(1, len(std)):
        a = np.array([similar[addTo]["avg"], std[i]])
        b = a.std(axis=0).max().item()
        if (b < config["barDirectionClassifierThreshold"]):
            similar[addTo]["avg"] = a.mean(axis=0)
            similar[addTo]["amount"] += 1
        else:
            similar.append({"avg": std[i], "amount": 1, "start": i})
            addTo += 1

    #return max(similar, key=lambda x: x["amount"])
    return similar

def sample_for_bar_var(sponsorArray, samples):
    var = []
    for i in samples:
        avg = sponsorArray[i]#.mean(axis=0)
        var.append(list(filter(lambda x: x["start"] == 0 or x["start"]+x["amount"] == avg.shape[0]-1, guess_bar_var(avg))))
    return var

def sample_for_bar_std(sponsorArray, samples):
    std = []
    for i in samples:
        avg = sponsorArray[i]#.mean(axis=0)
        std.append(list(filter(lambda x: x["start"] == 0 or x["start"]+x["amount"] == avg.shape[0]-1, guess_bar_std(avg))))
    return std

def preprocess_samples(samples, maxFrames):
    if any(x > 1 or x < 0 for x in samples):
        print("Samples aren't betwean 0 and 1 inclusive. Skipping")
        print("Samples: "+str(samples))
        return None
    samples = [int(x*maxFrames) for x in samples]
    samples.sort()
    return samples

def get_samples(maxFrames):
    samples = []
    while True:
        match config["sampleDirectionType"]:
            case "equidistant":
                samples = [x/(config["sampleForBarDirection"]+1) for x in range(1, config["sampleForBarDirection"]+1)]
                break
            case "random":
                random.seed(config["seed"])
                samples = [random.random() for x in range(config["sampleForBarDirection"])]
                break
            case "custom":
                if (config["sampleDirectionCustom"] == []):
                    config["sampleDirectionType"] = "equidistant"
                else:
                    samples = config["sampleDirectionCustom"]
                    break
            case _:
                print("Unsupported value '"+str(config["sampleDirectionType"])+"' for 'sampleDirectionType'. Continuing with value 'equidistant'")
                config["sampleDirectionType"] = "equidistant"
    return preprocess_samples(samples, maxFrames)

def guess_bar_direction(sponsorArray, samples, type):
    if (samples == None):
        return None
    match type:
        case "std":
            print("Guessing progress bar direction using standard deviation")
            potBar = sample_for_bar_std(sponsorArray, samples)
        case "var":
            print("Guessing progress bar direction using variance")
            potBar = sample_for_bar_var(sponsorArray, samples)
        case _:
            print("Invalid option for 'type'. Skipping")
            return None
    lengths = [len(x) for x in potBar]
    test = max(lengths)
    if (min(lengths) == 0):
        emptyCount = lengths.count(0)
        print("No left-to-right or right-to-left progress bar found "+str(emptyCount)+"/"+str(len(potBar))+" times")
        if (emptyCount/len(potBar) > config["noBarThresholdPercentage"]):
            print("Too few. Skipping")
            return None
        else:
            print("Enough found. Continuing")
    guess = None
    match test:
        case 1:
            potBar = [x[0] for x in potBar]
            potBarStart = list(filter(lambda x: x["start"] == 0, potBar))
            potBarEnd = list(filter(lambda x: x["start"]+x["amount"] == sponsorArray.shape[0]-1, potBar))
            if (abs(len(potBarStart)-len(potBarEnd)) < len(potBar)*config["minmumBarDifferencePercentage"]):
                print("Could be either left-to-right or right-to-left progress bar. Skipping")
                return None
            if (len(potBarStart) >= len(potBarEnd)):
                potBar = potBarStart
                guess = "left-to-right"
            else:
                potBar = potBarEnd
                guess = "right-to-left"
        case 2:
            potBarStart = [x for a in potBar for x in a if (x["start"] == 0)]
            potBarEnd = [x for a in potBar for x in a if (x["start"]+x["amount"] == sponsorArray.shape[0]-1)]
            if (abs(len(potBarStart)-len(potBarEnd)) < len(potBar)*config["minmumBarDifferencePercentage"]):
                print("Could be either left-to-right or right-to-left progress bar. Skipping")
                return None
            if (len(potBarStart) >= len(potBarEnd)):
                potBar = potBarStart
                guess = "left-to-right"
            else:
                potBar = potBarEnd
                guess = "right-to-left"
        case _:
            print("Uhh... Idk dude... I'm just gonna skip")
            print("Some stuff to diagnose:")
            print("  List of filtered lists of either left-to-right or right-to-left potential progress bars:")
            print("    "+str(potBar))
            print("  Longest sublist length (reason for failure):")
            print("    "+str(test))
            print("  List of longest lists:")
            print("    "+str([x for x in potBar if (len(x) == test)]))
            return None
    diff = []
    for i in range(len(potBar)-1):
        diff.append(potBar[i+1]["amount"]-potBar[i]["amount"])
    diff = [True if (x > 0) else False for x in diff]

    guess2 = "left-to-right" if (diff.count(True) > diff.count(False)) else "right-to-left"

    if (guess != guess2):
        print("Conflicting guesses")
        print("  First guess: "+guess)
        print("  Second guess: "+guess2)
        if (config["skipOnGuessConflict"]):
            print("Skipping")
            return None
        print("Continuing with second guess")
    else:
        print("Progress bar direction: "+guess2)
    return guess2

def sample_bar(sponsorArray, direction, type):
    match type:
        case "std":
            compFunc = lambda a: a.std(axis=0).max().item()
        case "var":
            compFunc = lambda a: a.var(axis=0).max().item()
        case _:
            print("Invalid option for 'type'. Skipping")
            return None
    match direction:
        case "left-to-right":
            indicies = list(range(sponsorArray.shape[1]))
        case "right-to-left":
            indicies = list(range(sponsorArray.shape[1]-1, -1, -1))
        case _:
            return None
    startIndex = indicies.pop(0)
    barPos = np.arange(sponsorArray.shape[0], dtype=np.uint8)
    for t in range(sponsorArray.shape[0]):
        frame = sponsorArray[t]
        start = frame[startIndex]
        pos = 0
        for i in indicies:
            compArry = np.array([start, frame[i]])
            if (compFunc(compArry) < 20):
                start = compArry.mean(axis=0)
                pos += 1
            else:
                break
        barPos[t] = pos
    return barPos

def plot_bar_positions(sponsorArray, barPos, video, sponsor):
    again = False
    plot = config["plot"]
    savedBarPositions = []
    barPositionsPlot = config["barPositionsPlot"]
    while True:
        match barPositionsPlot:
            case "percent":
                barPosProcessed = (barPos.astype(np.float16)/sponsorArray.shape[1])*100
                barStartFrame = np.arange(sponsorArray.shape[0])[barPosProcessed < config["startOfBarMaximumPercentage"]*100]
                if (barStartFrame.shape[0] == 0):
                    print("Couldn't find a start frame for the bar. Try adjusting 'startOfBarMaximumPercentage'. Assuming 0")
                    barStartFrame = 0
                else:
                    barStartFrame = barStartFrame[0].item()
                barEndFrame = np.arange(sponsorArray.shape[0])[barPosProcessed > config["endOfBarMinmumPercentage"]*100]
                if (barEndFrame.shape[0] == 0):
                    print("Couldn't find an end frame for the bar. Try adjusting 'endOfBarMaximumPercentage'. Assuming "+str(sponsorArray.shape[0]-1))
                    barEndFrame = sponsorArray.shape[0]-1
                else:
                    barEndFrame = barEndFrame[-1].item()
                maxPos = 100
                yLabel = "Progression in %"
            case "pixels":
                barPosProcessed = barPos
                barStartFrame = np.arange(sponsorArray.shape[0])[barPosProcessed < config["startOfBarMaximumPercentage"]*sponsorArray.shape[1]]
                if (barStartFrame.shape[0] == 0):
                    print("Couldn't find a start frame for the bar. Try adjusting 'startOfBarMaximumPercentage'. Assuming 0")
                    barStartFrame = 0
                else:
                    barStartFrame = barStartFrame[0].item()
                barEndFrame = np.arange(sponsorArray.shape[0])[barPosProcessed > config["endOfBarMinmumPercentage"]*sponsorArray.shape[1]]
                if (barEndFrame.shape[0] == 0):
                    print("Couldn't find an end frame for the bar. Try adjusting 'endOfBarMaximumPercentage'. Assuming "+str(sponsorArray.shape[0]-1))
                    barEndFrame = sponsorArray.shape[0]-1
                else:
                    barEndFrame = barEndFrame[-1].item()
                maxPos = sponsorArray.shape[1]
                yLabel = "Progression in pixels ("+str(sponsorArray.shape[1])+")"
            case "both":
                again = True
                barPositionsPlot = "percent"
                continue
            case _:
                return []

        if (plot["generatePlot"]):
            plt.title(plot["title"])
            plt.xlabel("Frames ("+str(barPosProcessed.shape[0])+")")
            plt.ylabel(yLabel)
            if (plot["highlightSponsorblockSponsorDefinition"]):
                plt.fill_between([0, barPosProcessed.shape[0]-1], maxPos, color=plot["SponsorblockSponsorDefinitionColour"], label="Sponsor segment defined by SponsorBlock")
            if (plot["highlightYouTuberSponsorDefinition"]):
                plt.fill_between([barStartFrame, barEndFrame], maxPos, color=plot["YouTuberSponsorDefinitionColour"], label="Sponsor segment defined by YouTuber")
            if (plot["showBarPositions"]):
                plt.plot(barPosProcessed, color="blue", label="Progress of the progress bar")
            plt.legend()
            plt.margins(0)
            if (plot["savePlot"]):
                plt.savefig(video["directory"]+"/graph"+str(sponsor["index"])+"_"+barPositionsPlot+"."+plot["savePlotFileExtension"])
            if (plot["showPlot"]):
                plt.show()
            plt.clf()

        if (config["saveBarPositions"] or config["saveBarInfo"]):
            path = video["directory"]+"/bar_positions"+str(sponsor["index"])+"_"+barPositionsPlot
            np.save(path, barPosProcessed)
            savedBarPositions.append(path+".npy")

        if (again):
            barPositionsPlot = "pixels"
            again = False
        else:
            return savedBarPositions

for video in videos:
    for sponsor in video["sponsor"]:
        print()
        print("Processing:", sponsor["path"])
        try:
            out, _ = (
                ffmpeg
                .input(sponsor["path"])
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, capture_stderr=True, quiet=config["ffmpegQuiet"])
            )
        except ffmpeg.Error as e:
            print("Error in ffmpeg")
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            print("Skipping")
            continue

        probe = ffmpeg.probe(sponsor["path"])["streams"][0]
        video["sponsorSize"]["height"] = probe["height"]
        video["sponsorSize"]["width"] = probe["width"]
        sponsor.update({"frames": int(probe["nb_frames"])})

        height = video["sponsorSize"]["height"]
        width = video["sponsorSize"]["width"]

        sponsorArray = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, height, width, 3])
        )

        samples = get_samples(sponsorArray.shape[0])

        sponsorArray = sponsorArray.mean(axis=1)

        direction = guess_bar_direction(sponsorArray, samples, config["barDirectionComparisonFunction"])
        sponsor.update({"direction": direction})

        barPos = sample_bar(sponsorArray, direction, config["sampleBarComparisonFunction"])

        if barPos is not None:
            savedBarPositions = plot_bar_positions(sponsorArray, barPos, video, sponsor)
            sponsor.update({"savedBarPositions": savedBarPositions})

    if (config["saveBarInfo"]):
        del video["video"]
        info = json.dumps(video, indent=2)
        with open(video["directory"]+"/barInfo.json", "w") as outfile:
            outfile.write(info)

print("Finished!")
