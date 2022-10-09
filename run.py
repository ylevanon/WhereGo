import os
from flask import Flask, render_template, request, redirect, url_for, session
from redis import Redis
#from flask.ext.sessions import Session
import os
import geocoder
import folium
import tasks
from rq import Queue
from werkzeug.debug import DebuggedApplication
from rq import Queue
from worker import conn
import urllib.parse

# Setup Flask
app = Flask(__name__)
'''
# Setting up session
#SESSION_TYPE = 'redis'
#app.config.from_object(__name__)
#Session(app)
'''

# Setup Redis
q = Queue(connection=conn)
#redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
#conn = redis.from_url(redis_url)


#redis_url = os.getenv('REDISTOGO_URL')

#urllib.parse.uses_netloc.append('redis')
#url = urllib.parse.urlparse(redis_url)
#conn = Redis(host=url.hostname, port=23070, db=0, password=url.password)
'''
# refactor the single quote marks
template_str=<html>
    <head>
      {% if refresh %}
        <meta http-equiv="refresh" content="5">
      {% endif %}
    </head>
    <body>{{result}}</body>
    </html>

def get_template(data, refresh=False):
    return render_template_string(template_str, result=data, refresh=refresh)
'''

def get_template(data, refresh=False):
    return render_template('loading_screen.html', result=data, refresh=refresh)


@app.route("/")
def index():
    
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template('about.html') 


@app.route("/map")
def map():
    return render_template('map.html')

'''
@app.route("/preference", methods=["POST"])
def preference():
    address = request.form['address']
    print(address)
    location = geocoder.osm(address)
    if  location.lat is None:
        error_statement = "Please enter a valid address."
        return render_template("index.html", error_statement=error_statement)
    latitude = location.lat
    longitude = location.lng
    my_map = folium.Map(location=(latitude, longitude), zoom_start=100, width='100%', height='55%')
    iframe = folium.IFrame(address, width=100, height=50)
    popup = folium.Popup(iframe, max_width=200)
    folium.Marker([latitude, longitude], popup=popup).add_to(my_map)
    my_map.save('project/templates/gmap.html')
    return render_template('preference.html', address=address, latitude=latitude, longitude=longitude)
'''
@app.route("/preference", methods=["GET","POST"])
def preference():
    if request.method == 'POST':
        address = request.form['address']
        print(address)
        location = geocoder.osm(address)
        if  location.lat is None:
            error_statement = "Please enter a valid address."
            return render_template("index.html", error_statement=error_statement)
        latitude = location.lat
        longitude = location.lng
        my_map = folium.Map(location=(latitude, longitude), zoom_start=100, width='100%', height='55%')
        iframe = folium.IFrame(address, width=100, height=50)
        popup = folium.Popup(iframe, max_width=200)
        folium.Marker([latitude, longitude], popup=popup).add_to(my_map)
        return render_template('preference.html', map_html=my_map._repr_html_(), address=address, latitude=latitude, longitude=longitude)

@app.route('/process', methods=['POST'])
def process():
    distance = request.form.get("distance", type=float)
    print(distance)
    latitude = request.form.get("latitude", type=float)
    longitude = request.form.get("longitude", type=float)
    address = request.form.get("address",type = str)
    print("address")
    #q = Queue(connection=conn)
    job = q.enqueue(tasks.model_builder, latitude, longitude, distance, address)
    return redirect(url_for('result', id=job.id))

@app.route('/checkstatus/<string:id>')
def result(id):
    print('id of the job printed above.')
    print(id)
    job = q.fetch_job(id)
    status = job.get_status()
    if status in ['queued', 'started', 'deferred', 'failed']:
        return get_template(status, refresh=True)
    elif status == 'finished':
        result = job.result 
        # If this is a string, we can simply return it:
        G = result[0]
        final_tour = result[1]
        distance = result[2]
        lat = result[3]
        print("latitude")
        print(lat)
        lng = result[4]
        print("lingitude")
        print(lng)
        address = result[5]
        print('address')
        print(address)
        run_map = folium.Map(location=(lat, lng), zoom_start=100, width='100%', height='55%')
        run_map = tasks.map_creation(G, final_tour, run_map)
        iframe = folium.IFrame(address, width=100, height=50)
        popup = folium.Popup(iframe, max_width=200)
        folium.Marker([lat, lng], popup=popup).add_to(run_map)
        return render_template('run_route.html',run_html=run_map._repr_html_(), distance=distance)

'''
@app.route("/run_route", methods=["POST"])
def route():
    street_crossing = request.form.get("street_crossing")
    distance = request.form.get("distance", type=float)
    print(distance)
    latitude = request.form.get("latitude", type=float)
    longitude = request.form.get("longitude", type=float)
    address = request.form.get("address",type = str)
    print("address")
    job = q.enqueue(tasks.model_builder, latitude, longitude, distance, address)
    return redirect(url_for('result', id=job.id))
    print(address)
    my_tup = tasks.model_builder(latitude, longitude, distance, address)
    return render_template(my_tup[0], distance=my_tup[1])
'''


if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    #q = Queue(connection=conn)
    app.run(debug=False, host="0.0.0.0", port=port)