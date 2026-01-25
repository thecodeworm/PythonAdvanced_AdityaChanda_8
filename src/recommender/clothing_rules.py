"""
Clothing Recommendation Engine
Provides clothing suggestions based on winter severity and weather conditions.
"""

from typing import Dict, List


def get_clothing_recommendations(severity: int, 
                                 temp: float = None,
                                 wind_chill: float = None,
                                 snow: float = None) -> Dict[str, List[str]]:
    """
    Get clothing recommendations based on severity and weather conditions.
    """
    effective_temp = wind_chill if wind_chill is not None else temp
    
    recommendations = {
        'head': [],
        'upper_body': [],
        'lower_body': [],
        'hands': [],
        'feet': [],
        'accessories': [],
        'activity_advice': []
    }
    
    if severity == 0:  # Mild
        recommendations['head'] = ["Light beanie or no hat needed"]
        recommendations['upper_body'] = ["Light jacket or hoodie", "Long-sleeve shirt", "Optional: Light sweater"]
        recommendations['lower_body'] = ["Regular pants or jeans", "Normal socks"]
        recommendations['hands'] = ["Optional: Light gloves"]
        recommendations['feet'] = ["Regular shoes or boots"]
        recommendations['accessories'] = ["Optional: Light scarf"]
        recommendations['activity_advice'] = ["Normal outdoor activities safe", "Good day for a walk or jog"]

    elif severity == 1:  # Snowy
        recommendations['head'] = ["Warm winter hat or beanie", "Optional: Ear warmers"]
        recommendations['upper_body'] = ["Insulated winter jacket", "Warm sweater or fleece underneath", "Long-sleeve base layer"]
        recommendations['lower_body'] = ["Warm pants or jeans", "Thermal/wool socks", "Optional: Long underwear if very cold"]
        recommendations['hands'] = ["Insulated gloves or mittens"]
        recommendations['feet'] = ["Waterproof winter boots", "Good tread for snow/ice"]
        recommendations['accessories'] = ["Warm scarf", "Optional: Neck gaiter"]
        recommendations['activity_advice'] = ["Use caution when walking outdoors", "Allow extra time for travel", "Watch for slippery surfaces", "Dress in layers"]
        
        if effective_temp is not None and effective_temp < 15:
            recommendations['upper_body'].append("Consider extra layer - very cold")
            recommendations['hands'] = ["Insulated mittens (warmer than gloves)"]

    elif severity == 2:  # Severe
        recommendations['head'] = ["Heavy winter hat covering ears", "Consider: Balaclava or face mask", "Protect all exposed skin"]
        recommendations['upper_body'] = ["Heavy parka or winter coat", "Multiple layers underneath: Thermal base layer, Fleece or wool middle layer, Insulated outer layer", "Ensure coat is waterproof"]
        recommendations['lower_body'] = ["Insulated snow pants or heavy pants with thermal underwear", "Thick wool or thermal socks", "Consider: Double sock layer"]
        recommendations['hands'] = ["Heavy insulated mittens", "Consider: Hand warmers inside", "Keep spare gloves dry"]
        recommendations['feet'] = ["Insulated winter boots", "Waterproof and rated for extreme cold", "Thick treaded soles for ice/snow"]
        recommendations['accessories'] = ["Heavy scarf or neck gaiter", "Consider: Face covering", "Goggles or sunglasses (snow glare)", "Hand/foot warmers"]
        recommendations['activity_advice'] = ["Avoid unnecessary outdoor exposure", "Limit time outside to under 30 minutes", "Travel only if absolutely necessary", "Risk of frostbite on exposed skin", "Stay indoors if possible"]
        
        if effective_temp is not None and effective_temp < -10:
            recommendations['activity_advice'].insert(0, "Extreme cold warning - Frostbite possible in minutes!")
        
        if snow is not None and snow > 4:
            recommendations['activity_advice'].insert(0, "Heavy snow warning - Roads may be impassable!")
    
    return recommendations


def format_recommendations(recommendations: Dict[str, List[str]], 
                           severity_name: str = None,
                           temp: float = None,
                           snow: float = None) -> str:
    """
    Format recommendations as a readable string.
    """
    output = []
    
    if severity_name:
        output.append(f"{'='*60}")
        output.append(f"WINTER CONDITIONS: {severity_name.upper()}")
        output.append(f"{'='*60}")
        if temp is not None:
            output.append(f"Temperature: {temp:.1f}°F")
        if snow is not None:
            output.append(f"Snowfall: {snow:.1f} inches")
        output.append("")
    
    categories = {
        'head': 'Head & Ears',
        'upper_body': 'Upper Body',
        'lower_body': 'Lower Body',
        'hands': 'Hands',
        'feet': 'Feet',
        'accessories': 'Accessories',
        'activity_advice': 'Activity Advice'
    }
    
    for key, title in categories.items():
        if recommendations.get(key):
            output.append(f"\n{title}:")
            for item in recommendations[key]:
                output.append(f"  • {item}")
    
    return '\n'.join(output)


def get_quick_recommendation(snow: float, temp: float, wind: float = 10.0) -> str:
    """
    Get a quick one-line clothing recommendation.
    """
    if temp <= 50 and wind > 3:
        wind_chill = 35.74 + 0.6215*temp - 35.75*(wind**0.16) + 0.4275*temp*(wind**0.16)
    else:
        wind_chill = temp
    
    if snow > 3.0 or ((snow > 1.5) and (wind > 20)) or (wind_chill < -10):
        return "Heavy winter gear required - parka, snow boots, face protection!"
    elif snow > 0.5 or ((snow > 0.1) and (temp < 25)):
        return "Warm winter clothing - insulated jacket, winter boots, gloves!"
    else:
        return "Light winter wear - jacket or hoodie, regular shoes OK"


def get_layering_guide(severity: int) -> List[str]:
    """
    Get layering guide for the severity level.
    """
    if severity == 0:
        return ["1 layer is usually sufficient", "Add a sweater if you get cold easily"]
    elif severity == 1:
        return ["Base Layer: Moisture-wicking thermal shirt",
                "Middle Layer: Fleece or wool sweater",
                "Outer Layer: Insulated winter jacket",
                "Tip: Layer up but not too tight - air provides insulation"]
    else:  # Severe
        return ["Base Layer: Thermal underwear top & bottom",
                "Middle Layer 1: Fleece or wool shirt",
                "Middle Layer 2: Insulated vest or thick sweater",
                "Outer Layer: Heavy waterproof parka",
                "Critical: NO cotton - stays wet and loses heat",
                "Tip: Dress warmer than you think you need"]


def get_time_limit_outside(severity: int, wind_chill: float = None) -> str:
    """
    Get recommended time limit for outdoor exposure.
    """
    if severity == 0:
        return "No time limit - conditions are safe"
    elif severity == 1:
        if wind_chill is not None and wind_chill < 10:
            return "Limit exposure to 1-2 hours, watch for cold symptoms"
        return "No strict limit, but dress appropriately and stay active"
    else:  # Severe
        if wind_chill is not None and wind_chill < -20:
            return "CRITICAL: Frostbite possible in 10 minutes or less!"
        elif wind_chill is not None and wind_chill < -10:
            return "Limit exposure to 20-30 minutes maximum"
        return "Limit exposure to 30-60 minutes, watch for frostbite"


if __name__ == "__main__":
    print("="*70)
    print("CLOTHING RECOMMENDATION ENGINE - TEST")
    print("="*70)
    
    scenarios = [
        {"name": "Clear Winter Day", "severity": 0, "severity_name": "Mild", "temp": 38, "wind_chill": 35, "snow": 0.0},
        {"name": "Snowy Morning", "severity": 1, "severity_name": "Snowy", "temp": 28, "wind_chill": 20, "snow": 2.0},
        {"name": "Blizzard Conditions", "severity": 2, "severity_name": "Severe", "temp": 15, "wind_chill": -5, "snow": 5.0}
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*70}")
        
        recs = get_clothing_recommendations(
            scenario['severity'],
            scenario['temp'],
            scenario['wind_chill'],
            scenario['snow']
        )
        
        print(format_recommendations(
            recs,
            scenario['severity_name'],
            scenario['temp'],
            scenario['snow']
        ))
        
        print(f"\nTime Limit Outside: {get_time_limit_outside(scenario['severity'], scenario['wind_chill'])}")
        print(f"\nLayering Guide:")
        for instruction in get_layering_guide(scenario['severity']):
            print(f"  • {instruction}")
