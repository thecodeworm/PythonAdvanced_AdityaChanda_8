import sys
import platform
from datetime import datetime, timedelta

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QGroupBox, QFrame, QGridLayout
)
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import Qt

# Mock recommendation & severity functions for demo
def predict_severity_from_snowfall(snow, temp, wind):
    severity = 0 if snow < 1 else 1 if snow < 2 else 2
    return {'severity': severity, 'name': ["Mild","Moderate","Severe"][severity], 'wind_chill': temp - wind*0.7}

def get_clothing_recommendations(severity, temp, wind_chill, snow):
    return {
        'upper_body': ["Jacket", "Coat", "Heavy Coat"],
        'feet': ["Shoes", "Boots", "Snow Boots"],
        'accessories': ["Hat", "Scarf", "Gloves"],
        'activity_advice': ["Safe", "Be careful", "Limit outdoor time"]
    }

class WinterPlannerGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Cross-platform emoji font
        system = platform.system()
        if system == "Windows":
            self.emoji_font_family = "Segoe UI Emoji"
        elif system == "Darwin":
            self.emoji_font_family = "Apple Color Emoji"
        else:
            self.emoji_font_family = "Noto Color Emoji"

        self.setWindowTitle("❄️ Smart Winter Planner")
        self.setGeometry(100, 100, 1200, 800)
        self.set_dark_theme()

        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addWidget(self.create_header())

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(30, 30, 30, 30)
        content_layout.setSpacing(20)

        # Generate forecast button
        forecast_btn = QPushButton("🔮  GENERATE 3-DAY FORECAST")
        forecast_btn.clicked.connect(self.generate_forecast)
        forecast_btn.setMinimumHeight(60)
        forecast_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        content_layout.addWidget(forecast_btn)

        # Forecast container
        self.forecast_container = QWidget()
        self.forecast_layout = QHBoxLayout(self.forecast_container)
        self.forecast_layout.setSpacing(20)
        content_layout.addWidget(self.forecast_container)
        self.forecast_container.hide()

        # Welcome screen
        self.welcome_widget = self.create_welcome()
        content_layout.addWidget(self.welcome_widget)

        main_layout.addWidget(content)
        self.statusBar().setStyleSheet("background-color: #1a1a1a; color: #888;")
        self.statusBar().showMessage("🟢 Ready")

    def set_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(40, 40, 40))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(40, 40, 40))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(66, 150, 250))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(66, 150, 250))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(dark_palette)

        style = """
        QPushButton { background: #4CAF50; color: white; border-radius: 8px; padding: 15px; font-weight: bold; }
        QPushButton:hover { background: #45a049; }
        QGroupBox { background-color: #2a2a2a; border: 2px solid #3a3a3a; border-radius: 12px; padding: 20px; }
        QLabel { color: #ddd; }
        QTextEdit { background-color: #252525; border: 1px solid #3a3a3a; border-radius: 8px; color: #ddd; }
        """
        self.setStyleSheet(style)

    def create_header(self):
        header = QWidget()
        header.setFixedHeight(120)
        header.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #2196F3, stop:0.5 #1976D2, stop:1 #0D47A1);
        """)
        layout = QVBoxLayout(header)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("❄️  SMART WINTER PLANNER")
        title.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: white; margin: 0;")
        layout.addWidget(title)

        subtitle = QLabel("AI-Powered 3-Day Snowfall Forecast & Clothing Recommendations")
        subtitle.setFont(QFont("Arial", 13))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.9); margin-top: 5px;")
        layout.addWidget(subtitle)

        return header

    def create_welcome(self):
        welcome = QWidget()
        layout = QVBoxLayout(welcome)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_label = QLabel("❄️")
        icon_label.setFont(QFont(self.emoji_font_family, 120))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        msg = QLabel("Click the button above to generate your 3-day winter forecast")
        msg.setFont(QFont("Arial", 16))
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet("color: #888; margin-top: 20px;")
        layout.addWidget(msg)

        features = QLabel("🔮 LSTM Neural Network  •  📊 Severity Classification  •  👔 Smart Recommendations")
        features.setFont(QFont(self.emoji_font_family, 12))
        features.setAlignment(Qt.AlignmentFlag.AlignCenter)
        features.setStyleSheet("color: #666; margin-top: 10px;")
        layout.addWidget(features)

        return welcome

    def create_day_card(self, day_num, prediction):
        date = prediction['date']
        snow = prediction['snowfall']
        temp = prediction['temp']
        wind = prediction['wind']

        severity_result = predict_severity_from_snowfall(snow, temp, wind)
        severity = severity_result['severity']
        severity_name = severity_result['name']
        wind_chill = severity_result['wind_chill']

        if severity == 0:
            color, bg_color, emoji = "#4CAF50", "#2e7d32", "🟢"
        elif severity == 1:
            color, bg_color, emoji = "#FFC107", "#F57C00", "🟡"
        else:
            color, bg_color, emoji = "#F44336", "#C62828", "🔴"

        card = QGroupBox()
        card.setMinimumWidth(350)
        card.setMaximumWidth(400)
        card.setStyleSheet(f"QGroupBox {{ background-color: #2a2a2a; border: 3px solid {color}; border-radius: 15px; padding: 20px; margin:0; }}")

        layout = QVBoxLayout(card)
        layout.setSpacing(15)

        day_header = QLabel(f"{emoji}  DAY {day_num}")
        day_header.setFont(QFont(self.emoji_font_family, 18, QFont.Weight.Bold))
        day_header.setStyleSheet(f"color: {color};")
        layout.addWidget(day_header)

        date_label = QLabel(date.strftime('%A, %B %d'))
        date_label.setFont(QFont("Arial", 14))
        date_label.setStyleSheet("color: #bbb;")
        layout.addWidget(date_label)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"background-color: {color}; max-height: 2px;")
        layout.addWidget(divider)

        weather_grid = QGridLayout()
        weather_grid.setSpacing(10)
        weather_items = [
            ("❄️", "Snowfall", f"{snow:.1f}\""),
            ("🌡️", "Temp", f"{temp}°F"),
            ("💨", "Wind", f"{wind} mph"),
            ("🥶", "Feels Like", f"{wind_chill:.0f}°F"),
        ]
        for i, (icon, label, value) in enumerate(weather_items):
            row = i // 2
            col = (i % 2) * 3
            icon_label = QLabel(icon)
            icon_label.setFont(QFont(self.emoji_font_family, 16))
            weather_grid.addWidget(icon_label, row, col)
            text_label = QLabel(f"{label}:\n{value}")
            text_label.setFont(QFont("Arial", 11))
            text_label.setStyleSheet("color: #ccc;")
            weather_grid.addWidget(text_label, row, col + 1)
        layout.addLayout(weather_grid)

        severity_badge = QLabel(f"   {severity_name.upper()}   ")
        severity_badge.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        severity_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        severity_badge.setStyleSheet(f"background-color: {bg_color}; color: white; border-radius: 15px; padding: 8px; margin:10px 0;")
        layout.addWidget(severity_badge)

        recs = get_clothing_recommendations(severity, temp, wind_chill, snow)
        tips_display = QTextEdit()
        tips_display.setReadOnly(True)
        tips_display.setFont(QFont(self.emoji_font_family, 12))
        tips_text = ""
        if recs.get('upper_body'):
            tips_text += f"🧥 {recs['upper_body'][0]}\n"
        if recs.get('feet'):
            tips_text += f"👢 {recs['feet'][0]}\n"
        if recs.get('accessories'):
            tips_text += f"🧣 {recs['accessories'][0]}"
        tips_display.setText(tips_text)
        tips_display.setStyleSheet("background-color: transparent; border: none; color: #aaa;")
        layout.addWidget(tips_display)

        if recs.get('activity_advice'):
            advice_label = QLabel("⚠️  " + recs['activity_advice'][0])
            advice_label.setFont(QFont("Arial", 9))
            advice_label.setStyleSheet("color: #888; margin-top:5px;")
            layout.addWidget(advice_label)

        layout.addStretch()
        return card

    def generate_forecast(self):
        self.welcome_widget.hide()
        self.forecast_container.show()
        for i in reversed(range(self.forecast_layout.count())): 
            self.forecast_layout.itemAt(i).widget().setParent(None)
        predictions = self.predict_next_3_days()
        for i, pred in enumerate(predictions, 1):
            card = self.create_day_card(i, pred)
            self.forecast_layout.addWidget(card)
        self.statusBar().showMessage("✅ Forecast generated successfully!")

    def predict_next_3_days(self):
        demo_forecasts = [
            {"snow": 0.8, "temp": 28, "wind": 12},
            {"snow": 2.1, "temp": 25, "wind": 18},
            {"snow": 0.3, "temp": 32, "wind": 8}
        ]
        predictions = []
        for i, forecast in enumerate(demo_forecasts):
            date = datetime.now() + timedelta(days=i+1)
            predictions.append({
                'date': date,
                'snowfall': forecast['snow'],
                'temp': forecast['temp'],
                'wind': forecast['wind']
            })
        return predictions

def main():
    app = QApplication(sys.argv)
    window = WinterPlannerGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
