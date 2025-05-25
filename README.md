# Missile Booster Simulation Web App

This Python-based web application simulates the performance of a multi-stage missile booster system. It allows users to input rocket parameters and outputs calculated results (such as thrust and delta-v) in both visual and Excel formats.

Originally developed as a standalone desktop app, the simulation runs on a local Flask server with a browser-based UI.

🔗 **GitHub Repo:** [Missile Booster Simulation](https://github.com/AChalli/missileBoosterCode-achalli-/tree/main)

## 🚀 Features

- Customizable booster stage configuration
- Real-time calculations of thrust and fuel consumption
- Export results to Excel for further analysis
- Responsive web interface built with HTML/CSS and Flask

## 🖥️ How to Use (End Users)

1. Double-click the executable to launch the application  
2. Wait for the terminal to open  
   - If the web app doesn't open automatically:
     - Open a browser
     - Navigate to `http://localhost:5000`
3. Enter rocket parameters in the web form  
4. Click **"Calculate and Export to Excel"**  
5. Choose where to save the Excel file  

### ✅ System Requirements

- Windows 10 or later  
- Any modern web browser (Chrome, Firefox, Edge)  

### 🛠 Troubleshooting

- If the web page doesn’t load:  
  - Ensure no other application is using port `5000`  
- Close the app by closing the browser or terminal window  

---

## 💻 Development Setup (For Reviewers)

1. Clone the repo:
   ```bash
   git clone https://github.com/AChalli/missileBoosterCode-achalli-.git
   cd missileBoosterCode-achalli-
