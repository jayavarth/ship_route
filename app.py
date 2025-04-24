from flask import Flask, jsonify, request
from optimization import nsga_ii_optimization, topsis_solution_ranking
from data_sources import fetch_weather_data, fetch_bunker_cost, fetch_vessel_data

app = Flask(__name__)

@app.route('/optimize-route', methods=['GET'])
def optimize_route():
    weather_data = fetch_weather_data(api_key="YOUR_API_KEY", latitude=20.0, longitude=80.0)
    bunker_cost = fetch_bunker_cost()
    vessel_data = fetch_vessel_data()

    pareto_solutions = nsga_ii_optimization(weather_data, bunker_cost, vessel_data)
    
    best_solution = topsis_solution_ranking(pareto_solutions)
    
    return jsonify(best_solution)

if __name__ == '__main__':
    app.run(debug=True)
