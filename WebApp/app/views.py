from django.shortcuts import render
import warnings
warnings.filterwarnings("ignore")
import folium
from folium import plugins
import requests
import io
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import pickle

# Create your views here.

def req(request):
    map1 = folium.Map([12, 122], tiles='CartoDB Dark_Matter', zoom_start=6)._repr_html_()
    context = {
        'map1': map1

    }
    return render(request, 'app/index.html', context)

def render_map(request):
    typhoon = request.POST.get('param')
    map1 = folium.Map([12, 122], tiles='CartoDB Dark_Matter', zoom_start=6)._repr_html_()
    if str(typhoon).lower() == "agathon":
        context = generate()
        print("foo")
    else:
        print("bar")
        context = {
            'message': "no typhoon found!",
            'map1': map1
        }
    return render(request, 'app/index.html', context)

def generate():
    url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/Agaton17k.csv"
    download = requests.get(url).content
    df = pd.read_csv(io.StringIO(download.decode('utf-8')))

    # Spatial analysis
    tweet_df_tl = spatial_analysis(df, "tl")
    tweet_df_en = spatial_analysis(df, "en")
    tweet_df = pd.concat([tweet_df_tl, tweet_df_en], axis=0).reset_index(drop=True)

    # Sentiment Analysis
    # Download trained model
    url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/pickle_model.pkl"
    file = requests.get(url,stream=True)
    logreg = pickle.loads(file.content)
    url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/vectorizer.pkl"
    file = requests.get(url,stream=True)
    vect = pickle.loads(file.content)
    # Predict sentiment values
    vectorized = vect.transform(tweet_df['cleaned'])
    pred = logreg.predict(vectorized)
    location_sentiment_df = pd.DataFrame({'coords': tweet_df['coords'], 'sentiment': pred})
    # Heatmap output
    heat_list = []
    for _loc in range(location_sentiment_df.shape[0]):
        temp = location_sentiment_df.coords[_loc].split("/")
        heat_list.append(
            [float(temp[0]), float(temp[1]), float(((location_sentiment_df.sentiment[_loc] * -1) + 2) / 4)])
    heat_list
    #data_list = Data.objects.values_list('latitude', 'longitude', 'value')
    map1 = folium.Map([12, 122],tiles='CartoDB Dark_Matter', zoom_start=6)
    plugins.HeatMap(heat_list).add_to(map1)
    map1 = map1._repr_html_()
    context={
        'map1':map1

    }
    return context


def spatial_analysis(df,LANGUAGE):
    if LANGUAGE == "tl":
        url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/tagalog_stopwords.csv"
        download = requests.get(url).content
        tagalog_stops = pd.read_csv(io.StringIO(download.decode('utf-8'))).a.to_list()
        stop_words = stopwords.words('english') + tagalog_stops
    else:
        stop_words = stopwords.words('english')
    temp = df[(df["Tweet Language"] == LANGUAGE)]

    def preprocessing_text(text):
        text = text.lower()
        text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text_tokens = word_tokenize(text)
        filtered_text = [w for w in text_tokens if not w in stop_words]
        return " ".join(filtered_text)

    tweet_df = pd.DataFrame()
    tweet_df['cleaned'] = temp['Tweet Content'].apply(preprocessing_text)
    tweet_df['coords'] = temp['Tweet Coordinates']
    tweet_df = tweet_df.reset_index(drop=True)
    # Get geojson data for cities in the Philippine
    url = "https://raw.githubusercontent.com/lorensdima/datasetstest/main/cities.geojson"
    f = requests.get(url)
    json_city = f.json()
    id_map = {}
    city_map = {}
    province_map = {}
    index = 0
    for feature in json_city["features"]:
        feature["id"] = index
        id_map[feature["properties"]["NAME_1"].lower() + " " + feature["properties"]["NAME_2"].lower()] = feature["id"]
        city_map[feature["properties"]["NAME_2"].lower()] = feature["id"]
        province_map[feature["properties"]["NAME_1"].lower()] = feature["id"]
        index += 1
    no_coords = tweet_df[tweet_df["coords"].isna()]
    with_coords = tweet_df.dropna(subset="coords")

    def locate_tweet(text):
        result = ""
        text_tokens = word_tokenize(text)
        city = ""
        province = ""
        init_loc = ""
        for w in text_tokens:
            if city_map.__contains__(w) and city == "":
                city = w
                init_loc = w + " " + init_loc
            if province_map.__contains__(w) and province == "":
                province = w
                init_loc = init_loc + w
            if city != "" and province != "":
                init_loc.strip()
                break;
        index_loc = 0
        if id_map.__contains__(init_loc):
            index_loc = id_map[init_loc]
        elif city != "":
            index_loc = city_map[city]
        elif province != "":
            index_loc = province_map[province]
        else:
            return np.NaN
        loc = json_city["features"][int(index_loc)]["geometry"]["coordinates"][0][0]
        location = re.findall('\\d+.\\d+', str(loc))

        return str(location[1]) + "/" + str(location[0])

    no_coords['coords'] = no_coords['cleaned'].apply(locate_tweet)
    no_coords = no_coords.dropna(subset="coords")

    def preprocess_coords(coords):
        numbers = re.findall('\\d+', coords)
        return numbers[2] + "." + numbers[3] + "/" + numbers[0] + "." + numbers[1]

    with_coords['coords'] = with_coords['coords'].apply(preprocess_coords)
    tweet_df = pd.concat([no_coords, with_coords], axis=0).reset_index(drop=True)
    return tweet_df