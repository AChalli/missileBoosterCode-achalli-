# ---------------------------------------------------------
# Missile Booster Web App
# Authored by AChalli with AI-assisted coding review.
# All core design and final implementation by AChalli.
# ---------------------------------------------------------

import numpy as np
from flask import Flask, render_template, request, send_file
from openpyxl import Workbook
import io
import threading
import webbrowser
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import base64
from io import BytesIO

app = Flask(__name__,
            template_folder="templates")

def parse_form(form):
    """
    Take a Flask form dict and return a kwargs dict
    ready for calculate_rocket_parameters().
    """
    return {
        'payload_kg':  250,
        'diameter_m':  0.75,
        'length_m':    7.5,
        'propellant_density':  float(form['propellant_density']),
        'isp':                float(form['isp']),
        'booster_lengths':    [float(form[f'booster{i}_length']) for i in (1,2,3)],
        'propellant_fractions':[float(form[f'booster{i}_fraction']) for i in (1,2,3)],
        'theta_deg':          float(form['launch_angle']),
        'mode':               form['mode'],
        'burn_time':          float(form.get('burn_time', 0))
    }

def build_excel_buffer(results):
    """
    Given the results dict from calculate_rocket_parameters,
    write an Excel workbook and return it as a BytesIO buffer.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Rocket Calculation Results"

    rows = [
        ("Stage 1 ΔV (m/s)", results['stage_delta_v'][0]),
        ("Stage 2 ΔV (m/s)", results['stage_delta_v'][1]),
        ("Stage 3 ΔV (m/s)", results['stage_delta_v'][2]),
        ("Total ΔV (m/s)",  results['total_delta_v']),
        ("Missile Range (km)", results['range_km']),
        ("Flight Time (s)",   results['flight_time']),
    ]

    for idx, (label, val) in enumerate(rows, start=1):
        ws[f"A{idx}"] = label
        ws[f"B{idx}"] = val

    buffer = io.BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer


def calculate_rocket_parameters(payload_kg, diameter_m, length_m,
                                propellant_density, isp,
                                booster_lengths, propellant_fractions,
                                theta_deg, mode='standard', burn_time=0):
    # Constants
    gravity_mps = 9.81
    radius_m = diameter_m / 2
    num_boosters = len(booster_lengths)

    # 1) volumes & propellant masses
    volumes = [np.pi * (radius_m**2) * h for h in booster_lengths]
    propellant_masses = [V * propellant_density for V in volumes]

    # 2) dry (structure) masses
    structural_masses = [
        (mp / propellant_fractions[i]) - mp
        for i, mp in enumerate(propellant_masses)
    ]
    structural_masses[-1] += payload_kg  # last‐stage carries the payload

    # 3) build up m0 & mf for each stage
    initial_masses = []
    final_masses   = []
    current_mass   = sum(propellant_masses) + sum(structural_masses)

    for i in range(num_boosters):
        # before burning stage i
        initial_masses.append(current_mass)
        # after burning stage i
        final_masses.append(initial_masses[-1] - propellant_masses[i])
        # drop stage i’s dry+propellant
        current_mass -= (propellant_masses[i] + structural_masses[i])

    # 4) ΔV per stage (special‐case first stage if in burntime mode)
    delta_v_stages = []
    for i in range(num_boosters):
        m0 = initial_masses[i]
        mf = final_masses[i]
        if i == 0 and mode == 'burntime':
            # V = t_b·g + Isp·g·ln(m0/mf)
            v = burn_time * gravity_mps + isp * gravity_mps * np.log(m0/mf)
        else:
            # classic rocket eqn
            v = isp * gravity_mps * np.log(m0/mf)
        delta_v_stages.append(v)

    total_delta_v = sum(delta_v_stages)

    # 5) range & flight time
    theta_rad = np.radians(theta_deg)
    range_km   = (total_delta_v**2 / gravity_mps) * np.sin(2*theta_rad) / 1000
    flight_time = 2 * total_delta_v * np.sin(theta_rad) / gravity_mps

    return {
        'stage_delta_v':    delta_v_stages,
        'total_delta_v':    total_delta_v,
        'range_km':         range_km,
        'flight_time':      flight_time,
        'initial_masses':   initial_masses,
        'propellant_masses':propellant_masses,
        'structural_masses':structural_masses
    }



def generate_trajectory_graph(v_max, theta_deg):
    """Generate x-y trajectory graph"""
    g = 9.81
    theta_rad = np.radians(theta_deg)

    # Calculate flight time
    flight_time = (2 * v_max * np.sin(theta_rad)) / g

    # Create time points
    t = np.linspace(0, flight_time, 500)

    # Calculate x and y components of trajectory
    v_x = v_max * np.cos(theta_rad)
    x = v_x * t
    y = v_max * np.sin(theta_rad) * t - 0.5 * g * t ** 2

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x / 1000, y / 1000)  # Convert to km
    plt.title('Missile Trajectory')
    plt.xlabel('Distance (km)')
    plt.ylabel('Altitude (km)')
    plt.grid(True)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Save plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_velocity_time_graph(v_max, theta_deg):
    """Generate velocity-time graph"""
    g = 9.81
    theta_rad = np.radians(theta_deg)

    # Calculate flight time
    flight_time = (2 * v_max * np.sin(theta_rad)) / g

    # Create time points
    t = np.linspace(0, flight_time, 500)

    # Calculate velocity components
    v_x = v_max * np.cos(theta_rad) * np.ones_like(t)
    v_y = v_max * np.sin(theta_rad) - g * t
    v_total = np.sqrt(v_x ** 2 + v_y ** 2)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, v_x, label='Horizontal Velocity')
    plt.plot(t, v_y, label='Vertical Velocity')
    plt.plot(t, v_total, label='Total Velocity')
    plt.title('Velocity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.legend()

    # Save plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def generate_stage_comparison_graph(delta_v_stages):
    """Generate stage comparison bar graph"""
    stages = [f"Stage {i + 1}" for i in range(len(delta_v_stages))]

    plt.figure(figsize=(8, 5))
    plt.bar(stages, delta_v_stages)
    plt.title('Delta-V Contribution by Stage')
    plt.xlabel('Rocket Stage')
    plt.ylabel('Delta-V (m/s)')
    plt.grid(True, axis='y')

    # Save plot to a bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def rocket_calculator():
    if request.method == 'POST':
        # 1) Safe parse of all inputs
        try:
            data = parse_form(request.form)
        except (KeyError, ValueError):
            # Missing field or non-numeric entry
            return render_template(
                'index.html',
                error="Please enter valid numbers in all fields."
            )

        # 2) Booster-length vs missile-length check
        total_length = sum(data['booster_lengths'])
        if abs(total_length - data['length_m']) > 1e-3:
            return render_template(
                'index.html',
                error=(
                  f"Stage lengths sum to {total_length:.2f} m, "
                  f"but missile length is {data['length_m']:.2f} m."
                )
            )

        # 3) Run calculation
        results = calculate_rocket_parameters(**data)

        # 4) Generate graphs using first-stage ΔV
        trajectory_img = generate_trajectory_graph(
            results['stage_delta_v'][0],
            data['theta_deg']
        )
        velocity_time_img = generate_velocity_time_graph(
            results['stage_delta_v'][0],
            data['theta_deg']
        )
        stage_comparison_img = generate_stage_comparison_graph(
            results['stage_delta_v']
        )

        # 5) Render results page
        return render_template(
            'results.html',
            results=results,
            trajectory_img=trajectory_img,
            velocity_time_img=velocity_time_img,
            stage_comparison_img=stage_comparison_img
        )

    # GET request → show form
    return render_template('index.html')

@app.route('/download-excel', methods=['POST'])
def download_excel():
    # Parse inputs & calculate
    data = parse_form(request.form)
    results = calculate_rocket_parameters(**data)

    # Build the Excel buffer
    buffer = build_excel_buffer(results)

    return send_file(
        buffer,
        as_attachment=True,
        download_name='rocket_calculations.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


def open_browser():
    webbrowser.open('http://127.0.0.1:5000')


if __name__ == '__main__':
    # Start browser after a short delay
    threading.Timer(1.5, open_browser).start()

    # Run the Flask app
    app.run(port=5000)
