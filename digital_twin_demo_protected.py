"""
REAL-TIME DIGITAL TWIN-BASED PROPERTY VALUATION SYSTEM
Patent Demonstration & Functional Prototype - PROTECTED VERSION

CONFIDENTIAL: For patent filing discussion and authorized viewers only.

This system implements the novel invention disclosed in Patent Application:
"System and Method for Real-Time Digital Twin-Based Property Valuation and Verification"

Access controlled by password authentication.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import pandas as pd
from collections import deque
import time
import hashlib
from datetime import datetime
from scipy import signal
from scipy.stats import zscore
import json

# ═══════════════════════════════════════════════════════════════════════════
# AUTHENTICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

def check_password():
    """
    Password protection for patent confidentiality.
    
    IMPORTANT: Change these credentials before deployment!
    """
    
    # Define authorized users (username: password_hash)
    # Password is hashed with SHA-256 for basic security
    AUTHORIZED_USERS = {
        "ipflair": hashlib.sha256("Digital2025!".encode()).hexdigest(),
        "demo": hashlib.sha256("PatentDemo2025".encode()).hexdigest(),
        "investor": hashlib.sha256("RealEstate2025".encode()).hexdigest()
    }
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        username = st.session_state.get("username", "").strip().lower()
        password = st.session_state.get("password", "")
        
        if username in AUTHORIZED_USERS:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if password_hash == AUTHORIZED_USERS[username]:
                st.session_state["password_correct"] = True
                st.session_state["authenticated_user"] = username
                del st.session_state["password"]  # Don't store password
                return
        
        st.session_state["password_correct"] = False
    
    # Check if already authenticated
    if st.session_state.get("password_correct", False):
        return True
    
    # Show login form
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1>🔐 Digital Twin Property Valuation System</h1>
            <h3>Patent Demonstration - Protected Access</h3>
            <p style='color: #666; margin-top: 1rem;'>
                This demonstration is confidential and protected for patent filing purposes.<br/>
                Please enter your credentials to access the system.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.text_input("Username", key="username", placeholder="Enter username")
            st.text_input("Password", type="password", key="password", placeholder="Enter password")
            submit = st.form_submit_button("Access System", use_container_width=True)
            
            if submit:
                password_entered()
        
        # Show error if authentication failed
        if st.session_state.get("password_correct") == False:
            st.error("❌ Invalid username or password. Please try again.")
        
        st.markdown("---")
        st.info("""
            **For Access:**
            - Patent filing discussion: Contact patent holder
            - Investor access: Available after patent filing
            
            **Need credentials?** Contact the system administrator.
        """)
        
        # Add disclaimer
        st.markdown("""
            <div style='text-align: center; margin-top: 2rem; padding: 1rem; 
                        background-color: #f8f9fa; border-radius: 5px; font-size: 0.85rem;'>
                <strong>CONFIDENTIAL & PROPRIETARY</strong><br/>
                Patent Pending. Unauthorized access, use, or distribution prohibited.<br/>
                © 2025 Digital Twin Valuation System. All rights reserved.
            </div>
        """, unsafe_allow_html=True)
    
    return False


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION (Must be first Streamlit command)
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Digital Twin Property Valuation - PROTECTED",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Patent Demonstration: Real-Time Digital Twin Property Valuation System (CONFIDENTIAL)"
    }
)

# Check authentication before proceeding
if not check_password():
    st.stop()

# Display authenticated user in sidebar
with st.sidebar:
    st.success(f"✓ Authenticated as: **{st.session_state['authenticated_user']}**")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["password_correct"] = False
        st.session_state["authenticated_user"] = None
        st.rerun()
    st.markdown("---")

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .patent-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .confidential-badge {
        background-color: #ffd700;
        color: #000;
        padding: 0.3rem 0.8rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .algorithm-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# ALL SYSTEM CLASSES (Same as before)
# ═══════════════════════════════════════════════════════════════════════════

class IoTSensorNetwork:
    """IoT Sensor Network - Element B/B1"""
    
    def __init__(self):
        self.time_step = 0
        self.base_noise_level = 0.02
        self.drift_accumulator = {
            'vibration': 0, 'strain': 0, 'moisture': 0, 'temperature': 0,
            'occupancy': 0, 'electrical': 0, 'air_quality': 0
        }
        
    def add_realistic_noise(self, signal, noise_level=0.02):
        return signal + np.random.normal(0, noise_level)
    
    def simulate_vibration(self, t):
        base_sway = 0.05 * np.sin(0.1 * t)
        high_freq = 0.02 * np.sin(5 * t)
        event = 0.15 if (t % 50) < 2 else 0
        drift = self.drift_accumulator['vibration']
        signal = abs(base_sway + high_freq + event + drift)
        return self.add_realistic_noise(signal, 0.01)
    
    def simulate_strain(self, t):
        daily_cycle = 0.3 * np.sin(0.5 * t)
        seasonal = 0.1 * np.sin(0.05 * t)
        fatigue = min(0.2, t * 0.0001)
        signal = 0.4 + daily_cycle + seasonal + fatigue
        return max(0, min(1, self.add_realistic_noise(signal, 0.03)))
    
    def simulate_moisture(self, t):
        seasonal_humidity = 0.5 + 0.3 * np.sin(0.03 * t)
        weather_event = 0.2 if (t % 30) < 5 else 0
        leak_risk = 0.1 if random.random() < 0.02 else 0
        signal = seasonal_humidity + weather_event + leak_risk
        return max(0, min(1, self.add_realistic_noise(signal, 0.04)))
    
    def simulate_temperature(self, t):
        daily_cycle = 0.3 * np.sin(0.4 * t)
        seasonal_cycle = 0.4 * np.sin(0.02 * t)
        hvac_efficiency = -0.1 * (t * 0.0001)
        signal = 0.5 + daily_cycle + seasonal_cycle + hvac_efficiency
        return max(0, min(1, self.add_realistic_noise(signal, 0.02)))
    
    def simulate_occupancy(self, t):
        hour_of_day = (t % 24)
        is_work_hours = 1.0 if 9 <= hour_of_day <= 18 else 0.2
        weekly_pattern = 0.8 if (t % 168) < 120 else 0.3
        signal = is_work_hours * weekly_pattern * (0.6 + 0.4 * np.sin(1.3 * t))
        return max(0, min(1, self.add_realistic_noise(signal, 0.05)))
    
    def simulate_electrical_load(self, t):
        occupancy_factor = self.simulate_occupancy(t)
        baseline_load = 0.2
        peak_load = 0.6 * occupancy_factor
        equipment_aging = min(0.15, t * 0.0002)
        signal = baseline_load + peak_load + equipment_aging
        return max(0, min(1, self.add_realistic_noise(signal, 0.03)))
    
    def simulate_air_quality(self, t):
        occupancy = self.simulate_occupancy(t)
        ventilation_quality = 1.0 - (t * 0.0001)
        external_pollution = 0.3 * np.sin(0.1 * t)
        signal = 0.4 + (0.4 * occupancy) - (0.3 * ventilation_quality) + external_pollution
        return max(0, min(1, self.add_realistic_noise(signal, 0.04)))
    
    def get_all_sensors(self, t):
        self.time_step = t
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'vibration': self.simulate_vibration(t),
            'strain': self.simulate_strain(t),
            'moisture': self.simulate_moisture(t),
            'temperature': self.simulate_temperature(t),
            'occupancy': self.simulate_occupancy(t),
            'electrical_load': self.simulate_electrical_load(t),
            'air_quality': self.simulate_air_quality(t)
        }


class DataPreprocessor:
    """Data Preprocessing Pipeline - Element C/C1"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.buffers = {}
        
    def denoise_savgol(self, data, window=5, polyorder=2):
        if len(data) < window:
            return data
        try:
            return signal.savgol_filter(data, window, polyorder)
        except:
            return data
    
    def normalize_zscore(self, data):
        if len(data) < 2:
            return data
        try:
            return zscore(data)
        except:
            return data
    
    def moving_average(self, data, window=5):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def detect_outliers_iqr(self, data):
        if len(data) < 4:
            return np.zeros(len(data), dtype=bool)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers
    
    def align_timestamps(self, sensor_data):
        return sensor_data
    
    def preprocess_stream(self, sensor_name, value, history):
        if sensor_name not in self.buffers:
            self.buffers[sensor_name] = deque(maxlen=self.window_size)
        
        self.buffers[sensor_name].append(value)
        buffer_data = np.array(self.buffers[sensor_name])
        
        outliers = self.detect_outliers_iqr(buffer_data)
        if outliers.any():
            median_val = np.median(buffer_data[~outliers])
            buffer_data[outliers] = median_val
        
        if len(buffer_data) >= 5:
            denoised = self.denoise_savgol(buffer_data, window=5, polyorder=2)
            processed_value = denoised[-1]
        else:
            processed_value = buffer_data[-1]
        
        return processed_value


class DigitalTwinModel:
    """Digital Twin Model - Element D/D1"""
    
    def __init__(self, property_id="PROP-001"):
        self.property_id = property_id
        self.creation_time = datetime.now()
        self.current_state = {
            'structural_integrity': 1.0,
            'environmental_stability': 1.0,
            'operational_efficiency': 1.0,
            'maintenance_status': 1.0,
            'age_factor': 0.0
        }
        self.history_window = 100
        self.state_history = {
            'structural': deque(maxlen=self.history_window),
            'environmental': deque(maxlen=self.history_window),
            'usage': deque(maxlen=self.history_window),
            'maintenance': deque(maxlen=self.history_window)
        }
        self.update_count = 0
        self.last_update = None
        self.anomaly_flags = []
        
    def update_state(self, sensor_data, technical_indicators):
        alpha = 0.3
        structural_score = 1.0 - (0.5 * sensor_data['vibration'] + 0.5 * sensor_data['strain'])
        self.current_state['structural_integrity'] = (
            alpha * structural_score + (1 - alpha) * self.current_state['structural_integrity']
        )
        env_score = 1.0 - (0.4 * sensor_data['moisture'] + 0.3 * sensor_data['temperature'] + 
                          0.3 * sensor_data['air_quality'])
        self.current_state['environmental_stability'] = (
            alpha * env_score + (1 - alpha) * self.current_state['environmental_stability']
        )
        ops_score = 1.0 - (0.6 * sensor_data['electrical_load'] + 0.4 * sensor_data['air_quality'])
        self.current_state['operational_efficiency'] = (
            alpha * ops_score + (1 - alpha) * self.current_state['operational_efficiency']
        )
        self.state_history['structural'].append(self.current_state['structural_integrity'])
        self.state_history['environmental'].append(self.current_state['environmental_stability'])
        self.state_history['usage'].append(sensor_data['occupancy'])
        self.update_count += 1
        self.last_update = datetime.now()
    
    def detect_anomalies(self):
        anomalies = []
        for metric, history in self.state_history.items():
            if len(history) < 20:
                continue
            data = np.array(history)
            mean = np.mean(data)
            std = np.std(data)
            upper_limit = mean + 3 * std
            lower_limit = mean - 3 * std
            current_value = data[-1]
            if current_value > upper_limit or current_value < lower_limit:
                anomalies.append({
                    'metric': metric,
                    'value': current_value,
                    'expected_range': (lower_limit, upper_limit),
                    'severity': 'HIGH' if abs(current_value - mean) > 4 * std else 'MEDIUM'
                })
        return anomalies
    
    def get_trend_analysis(self):
        trends = {}
        for metric, history in self.state_history.items():
            if len(history) < 10:
                trends[metric] = {'slope': 0, 'status': 'INSUFFICIENT_DATA'}
                continue
            data = np.array(history)
            x = np.arange(len(data))
            slope = np.polyfit(x, data, 1)[0]
            if slope > 0.001:
                status = 'IMPROVING'
            elif slope < -0.001:
                status = 'DEGRADING'
            else:
                status = 'STABLE'
            trends[metric] = {
                'slope': slope,
                'status': status,
                'current': data[-1],
                'average': np.mean(data)
            }
        return trends


class TechnicalIndicatorsEngine:
    """Technical Indicators Engine - Element E/E1"""
    
    def __init__(self):
        self.shf_exponent = 2.0
        self.tau1 = 0.3
        self.tau2 = 0.7
        self.epsilon = 0.6
        self.gamma = 2.0
        self.alpha = 10.0
        self.theta = 0.5
        self.lambda_c = 5.0
        self.sensor_history = {
            'vibration': deque(maxlen=50),
            'environment': deque(maxlen=50),
            'usage': deque(maxlen=50)
        }
    
    def calculate_shf(self, vibration_normalized, strain_normalized):
        s = 0.6 * vibration_normalized + 0.4 * strain_normalized
        s = max(0.0, min(1.0, s))
        shf = 1.0 - (s ** self.shf_exponent)
        self.sensor_history['vibration'].append(s)
        return shf, s
    
    def calculate_esf(self, moisture, temperature, air_quality):
        e = 0.4 * moisture + 0.3 * temperature + 0.3 * air_quality
        e = max(0.0, min(1.0, e))
        if e < self.tau1:
            esf = 1.0
        elif e <= self.tau2:
            fraction = (e - self.tau1) / (self.tau2 - self.tau1)
            esf = 1.0 - fraction * (1.0 - self.epsilon)
        else:
            esf = self.epsilon * 0.8
        self.sensor_history['environment'].append(e)
        return esf, e
    
    def calculate_uss(self, occupancy, electrical_load):
        u = 0.5 * occupancy + 0.5 * electrical_load
        u = max(0.0, min(1.0, u))
        uss = 1.0 - (u ** self.gamma)
        self.sensor_history['usage'].append(u)
        return uss, u
    
    def calculate_pdp(self, property_age_normalized):
        p = max(0.0, min(1.0, property_age_normalized))
        try:
            pdp = 1.0 / (1.0 + math.exp(self.alpha * (p - self.theta)))
        except OverflowError:
            pdp = 0.0
        return pdp, p
    
    def calculate_confidence_index(self):
        if len(self.sensor_history['vibration']) < 5:
            return 1.0
        std_vibration = np.std(list(self.sensor_history['vibration']))
        std_environment = np.std(list(self.sensor_history['environment']))
        std_usage = np.std(list(self.sensor_history['usage']))
        sigma_avg = np.mean([std_vibration, std_environment, std_usage])
        ci = math.exp(-self.lambda_c * sigma_avg)
        return ci


class PropertyValuationEngine:
    """Property Valuation Engine - Element F/F1/F2"""
    
    def __init__(self, base_market_value, land_value):
        self.base_market_value = base_market_value
        self.land_value = land_value
        self.structure_value = base_market_value - land_value
        self.valuation_history = deque(maxlen=1000)
        self.factor_history = {
            'shf': deque(maxlen=1000),
            'esf': deque(maxlen=1000),
            'uss': deque(maxlen=1000),
            'pdp': deque(maxlen=1000),
            'ci': deque(maxlen=1000)
        }
        
    def calculate_rtpmv(self, shf, esf, uss, pdp, ci):
        health_factor = shf * esf * uss * pdp * ci
        adjusted_structure_value = self.structure_value * health_factor
        rtpmv = self.land_value + adjusted_structure_value
        self.valuation_history.append(rtpmv)
        self.factor_history['shf'].append(shf)
        self.factor_history['esf'].append(esf)
        self.factor_history['uss'].append(uss)
        self.factor_history['pdp'].append(pdp)
        self.factor_history['ci'].append(ci)
        return rtpmv, health_factor
    
    def get_valuation_breakdown(self, rtpmv, health_factor, shf, esf, uss, pdp, ci):
        structure_current = self.structure_value * health_factor
        depreciation_amount = self.structure_value - structure_current
        depreciation_pct = (depreciation_amount / self.structure_value) * 100
        return {
            'rtpmv': rtpmv,
            'base_market_value': self.base_market_value,
            'land_value': self.land_value,
            'structure_base': self.structure_value,
            'structure_current': structure_current,
            'depreciation_amount': depreciation_amount,
            'depreciation_percentage': depreciation_pct,
            'health_factor': health_factor,
            'factor_contributions': {
                'SHF': shf,
                'ESF': esf,
                'USS': uss,
                'PDP': pdp,
                'CI': ci
            }
        }


class VerificationSystem:
    """Verification System - Element G/G1"""
    
    def __init__(self):
        self.audit_trail = []
        self.token_registry = {}
        
    def generate_verification_report(self, property_id, sensor_data, technical_indicators, 
                                     valuation_data, timestamp):
        report = {
            'report_id': f"VR-{int(time.time() * 1000)}",
            'property_id': property_id,
            'timestamp': timestamp,
            'sensor_data': sensor_data,
            'technical_indicators': technical_indicators,
            'valuation': valuation_data,
            'version': '1.0'
        }
        report_json = json.dumps(report, sort_keys=True)
        report_hash = hashlib.sha256(report_json.encode()).hexdigest()
        report['hash'] = report_hash
        report['signature_method'] = 'SHA-256'
        self.audit_trail.append({
            'hash': report_hash,
            'timestamp': timestamp,
            'report_id': report['report_id']
        })
        return report
    
    def generate_valuation_token(self, rtpmv, health_factor, confidence_index, 
                                 property_id, timestamp):
        token = {
            'token_id': f"VT-{property_id}-{int(time.time() * 1000)}",
            'property_id': property_id,
            'rtpmv': rtpmv,
            'health_factor': health_factor,
            'confidence_index': confidence_index,
            'timestamp': timestamp,
            'valid_until': None,
            'token_standard': 'RTPV-1.0'
        }
        token_json = json.dumps(token, sort_keys=True)
        token_hash = hashlib.sha256(token_json.encode()).hexdigest()
        token['token_hash'] = token_hash
        self.token_registry[token['token_id']] = token
        return token
    
    def verify_token(self, token_id):
        if token_id not in self.token_registry:
            return {'valid': False, 'reason': 'Token not found'}
        token = self.token_registry[token_id]
        stored_hash = token['token_hash']
        token_copy = {k: v for k, v in token.items() if k != 'token_hash'}
        token_json = json.dumps(token_copy, sort_keys=True)
        computed_hash = hashlib.sha256(token_json.encode()).hexdigest()
        if stored_hash == computed_hash:
            return {'valid': True, 'token': token}
        else:
            return {'valid': False, 'reason': 'Hash mismatch - token tampered'}
    
    def get_audit_trail(self, limit=100):
        return self.audit_trail[-limit:]


class MarketIntegrationLayer:
    """Market Integration Layer - Element H/H1/H1.1"""
    
    def __init__(self, property_id):
        self.property_id = property_id
        self.current_listing_price = None
        self.bid_ask_spread = None
        self.market_status = 'INITIALIZING'
        self.trading_enabled = True
        self.minimum_bid = None
        self.maximum_ask = None
        self.tick_size = 1000
        self.signal_history = deque(maxlen=500)
        
    def publish_valuation_signal(self, rtpmv, confidence_index, health_factor, timestamp):
        signal = {
            'signal_id': f"SIG-{int(time.time() * 1000)}",
            'property_id': self.property_id,
            'rtpmv': rtpmv,
            'confidence_index': confidence_index,
            'health_factor': health_factor,
            'timestamp': timestamp,
            'signal_type': 'VALUATION_UPDATE'
        }
        self.signal_history.append(signal)
        return signal
    
    def calculate_listing_price_guidance(self, rtpmv, confidence_index, market_sentiment=0.5):
        sentiment_factor = (market_sentiment - 0.5) * 0.4
        adjustment = 1 + (confidence_index * sentiment_factor)
        listing_price = rtpmv * adjustment
        listing_price = round(listing_price / self.tick_size) * self.tick_size
        self.current_listing_price = listing_price
        return listing_price
    
    def update_bid_ask_bands(self, rtpmv, confidence_index, volatility=0.02):
        base_spread = 0.05
        spread_multiplier = (2 - confidence_index) * (1 + volatility)
        total_spread = base_spread * spread_multiplier
        bid_price = rtpmv * (1 - total_spread / 2)
        ask_price = rtpmv * (1 + total_spread / 2)
        bid_price = round(bid_price / self.tick_size) * self.tick_size
        ask_price = round(ask_price / self.tick_size) * self.tick_size
        self.bid_ask_spread = {
            'bid': bid_price,
            'ask': ask_price,
            'spread': ask_price - bid_price,
            'spread_percentage': (ask_price - bid_price) / rtpmv * 100
        }
        return self.bid_ask_spread
    
    def apply_market_controls(self, rtpmv, confidence_index, health_factor):
        controls = {
            'trading_enabled': True,
            'restrictions': [],
            'requirements': [],
            'circuit_breaker_active': False
        }
        if confidence_index < 0.5:
            controls['trading_enabled'] = False
            controls['restrictions'].append('LOW_CONFIDENCE: Automated trading suspended')
        elif confidence_index < 0.7:
            controls['restrictions'].append('MODERATE_CONFIDENCE: Manual approval required for large orders')
        if health_factor < 0.6:
            controls['requirements'].append('ASSET_CONDITION: Additional inspection required for transactions')
        elif health_factor < 0.8:
            controls['requirements'].append('ASSET_CONDITION: Buyer notification of condition metrics required')
        if len(self.signal_history) >= 10:
            recent_rtpmvs = [s['rtpmv'] for s in list(self.signal_history)[-10:]]
            price_volatility = np.std(recent_rtpmvs) / np.mean(recent_rtpmvs)
            if price_volatility > 0.1:
                controls['circuit_breaker_active'] = True
                controls['restrictions'].append('HIGH_VOLATILITY: Trading halted for 5 minutes')
        self.trading_enabled = controls['trading_enabled']
        self.market_status = 'ACTIVE' if controls['trading_enabled'] else 'RESTRICTED'
        self.minimum_bid = rtpmv * 0.7
        self.maximum_ask = rtpmv * 1.3
        return controls


class DigitalTwinOrchestrator:
    """System Orchestrator - Integrates all patent elements"""
    
    def __init__(self, property_id, base_market_value, land_value):
        self.property_id = property_id
        self.sensor_network = IoTSensorNetwork()
        self.preprocessor = DataPreprocessor()
        self.digital_twin = DigitalTwinModel(property_id)
        self.indicators_engine = TechnicalIndicatorsEngine()
        self.valuation_engine = PropertyValuationEngine(base_market_value, land_value)
        self.verification_system = VerificationSystem()
        self.market_layer = MarketIntegrationLayer(property_id)
        self.time_step = 0
        
    def process_cycle(self):
        self.time_step += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        raw_sensor_data = self.sensor_network.get_all_sensors(self.time_step)
        processed_data = {}
        for sensor, value in raw_sensor_data.items():
            if sensor != 'timestamp':
                processed_data[sensor] = self.preprocessor.preprocess_stream(sensor, value, [])
        processed_data['timestamp'] = timestamp
        shf, s_metric = self.indicators_engine.calculate_shf(
            processed_data['vibration'], processed_data['strain']
        )
        esf, e_metric = self.indicators_engine.calculate_esf(
            processed_data['moisture'], processed_data['temperature'], processed_data['air_quality']
        )
        uss, u_metric = self.indicators_engine.calculate_uss(
            processed_data['occupancy'], processed_data['electrical_load']
        )
        property_age = min(1.0, self.time_step / 1000.0)
        pdp, p_metric = self.indicators_engine.calculate_pdp(property_age)
        ci = self.indicators_engine.calculate_confidence_index()
        technical_indicators = {
            'SHF': shf, 'ESF': esf, 'USS': uss, 'PDP': pdp, 'CI': ci,
            'metrics': {'s': s_metric, 'e': e_metric, 'u': u_metric, 'p': p_metric}
        }
        self.digital_twin.update_state(processed_data, technical_indicators)
        rtpmv, health_factor = self.valuation_engine.calculate_rtpmv(shf, esf, uss, pdp, ci)
        valuation_breakdown = self.valuation_engine.get_valuation_breakdown(
            rtpmv, health_factor, shf, esf, uss, pdp, ci
        )
        verification_report = self.verification_system.generate_verification_report(
            self.property_id, processed_data, technical_indicators, valuation_breakdown, timestamp
        )
        valuation_token = self.verification_system.generate_valuation_token(
            rtpmv, health_factor, ci, self.property_id, timestamp
        )
        valuation_signal = self.market_layer.publish_valuation_signal(
            rtpmv, ci, health_factor, timestamp
        )
        listing_price = self.market_layer.calculate_listing_price_guidance(rtpmv, ci)
        bid_ask = self.market_layer.update_bid_ask_bands(rtpmv, ci)
        market_controls = self.market_layer.apply_market_controls(rtpmv, ci, health_factor)
        return {
            'timestamp': timestamp,
            'sensor_data': processed_data,
            'technical_indicators': technical_indicators,
            'valuation': valuation_breakdown,
            'verification': {
                'report_id': verification_report['report_id'],
                'report_hash': verification_report['hash'],
                'token_id': valuation_token['token_id'],
                'token_hash': valuation_token['token_hash']
            },
            'market': {
                'signal_id': valuation_signal['signal_id'],
                'listing_price': listing_price,
                'bid_ask': bid_ask,
                'controls': market_controls
            }
        }


# ═══════════════════════════════════════════════════════════════════════════
# STREAMLIT UI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Main Streamlit application"""
    
    # Header with confidential badge
    st.markdown(
        '<div class="main-header">🏢 Digital Twin Property Valuation Oracle<br/>'
        '<span class="patent-badge">PATENT DEMONSTRATION</span>'
        '<span class="confidential-badge">⚠️ CONFIDENTIAL</span></div>', 
        unsafe_allow_html=True
    )
    
    # Initialize session state
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = DigitalTwinOrchestrator(
            property_id="PROP-2025-001",
            base_market_value=10_000_000,
            land_value=6_000_000
        )
    if 'data_log' not in st.session_state:
        st.session_state.data_log = []
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    # Sidebar
    with st.sidebar:
        st.title("🎮 System Controls")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ START", use_container_width=True):
                st.session_state.running = True
        with col2:
            if st.button("⏹️ STOP", use_container_width=True):
                st.session_state.running = False
        
        if st.button("🔄 RESET SYSTEM", use_container_width=True):
            st.session_state.orchestrator = DigitalTwinOrchestrator(
                property_id="PROP-2025-001",
                base_market_value=10_000_000,
                land_value=6_000_000
            )
            st.session_state.data_log = []
            st.session_state.running = False
            st.rerun()
        
        st.markdown("---")
        st.subheader("📋 Patent Elements")
        st.markdown("""
        ✅ **Element B/B1**: IoT Sensors  
        ✅ **Element C/C1**: Preprocessing  
        ✅ **Element D/D1**: Digital Twin  
        ✅ **Element E/E1**: Indicators  
        ✅ **Element F**: Valuation  
        ✅ **Element G/G1**: Verification  
        ✅ **Element H**: Market Integration
        """)
        
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.info("""
        This system demonstrates the complete patent workflow for real-time 
        digital twin-based property valuation using IoT sensors and 
        cryptographic verification.
        
        **Status**: Patent Filing in Progress
        """)
        
        # Add confidentiality notice
        st.markdown("---")
        st.warning("""
        **⚠️ CONFIDENTIAL**
        
        This demonstration is proprietary and protected. Unauthorized 
        disclosure or use is prohibited.
        """)
    
    # Main content area
    if st.session_state.running:
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        details_placeholder = st.empty()
        table_placeholder = st.empty()
        
        while st.session_state.running:
            cycle_data = st.session_state.orchestrator.process_cycle()
            st.session_state.data_log.append(cycle_data)
            
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    rtpmv = cycle_data['valuation']['rtpmv']
                    st.metric("💰 RTPMV", f"${rtpmv:,.0f}", 
                             delta=f"{((rtpmv/10_000_000 - 1) * 100):.1f}%")
                with col2:
                    health = cycle_data['valuation']['health_factor']
                    st.metric("❤️ Health Factor", f"{health*100:.1f}%", 
                             delta=f"{((health - 1) * 100):.1f}%")
                with col3:
                    ci = cycle_data['technical_indicators']['CI']
                    st.metric("🎯 Confidence", f"{ci*100:.1f}%")
                with col4:
                    status = cycle_data['market']['controls']['trading_enabled']
                    st.metric("📊 Market Status", "ACTIVE" if status else "RESTRICTED")
            
            with chart_placeholder.container():
                if len(st.session_state.data_log) > 0:
                    # Get current cycle data
                    current = cycle_data
                    indicators = current['technical_indicators']
                    metrics = indicators['metrics']
                    
                    # Create figure with 6 subplots showing CURVES with RED DOT
                    fig = plt.figure(figsize=(14, 7))
                    
                    # X-axis for theoretical curves (0 to 1)
                    x_axis = np.linspace(0, 1, 100)
                    
                    # TOP ROW: Structural, Environmental, Usage
                    
                    # 1. STRUCTURAL HEALTH FACTOR (SHF) - Quadratic curve
                    ax1 = fig.add_subplot(2, 3, 1)
                    shf_curve = 1 - x_axis**2  # Theoretical SHF curve
                    ax1.plot(x_axis, shf_curve, 'b-', linewidth=2, label='SHF = 1 - s²')
                    ax1.plot([metrics['s']], [indicators['SHF']], 'ro', markersize=12)
                    ax1.set_title('Structural Health Factor (SHF)\nQuadratic Penalty', fontweight='bold')
                    ax1.set_xlabel('Vibration/Strain Metric (s)')
                    ax1.set_ylabel('SHF Value')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_xlim(0, 1)
                    ax1.set_ylim(0, 1.05)
                    ax1.text(0.02, 0.95, f'Current: {indicators["SHF"]:.3f}', 
                            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    
                    # 2. ENVIRONMENTAL STABILITY FACTOR (ESF) - Piecewise curve
                    ax2 = fig.add_subplot(2, 3, 2)
                    # Piecewise ESF curve
                    esf_curve = []
                    for e_val in x_axis:
                        if e_val < 0.3:
                            esf_curve.append(1.0)
                        elif e_val <= 0.7:
                            frac = (e_val - 0.3) / 0.4
                            esf_curve.append(1.0 - frac * 0.4)
                        else:
                            esf_curve.append(0.48)
                    ax2.plot(x_axis, esf_curve, 'g-', linewidth=2, label='ESF Piecewise')
                    ax2.plot([metrics['e']], [indicators['ESF']], 'ro', markersize=12)
                    ax2.set_title('Environmental Stability Factor (ESF)\nPiecewise Linear', fontweight='bold')
                    ax2.set_xlabel('Environment Metric (e)')
                    ax2.set_ylabel('ESF Value')
                    ax2.grid(True, alpha=0.3)
                    ax2.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, label='τ₁=0.3')
                    ax2.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5, label='τ₂=0.7')
                    ax2.set_xlim(0, 1)
                    ax2.set_ylim(0, 1.05)
                    ax2.text(0.02, 0.95, f'Current: {indicators["ESF"]:.3f}', 
                            transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                    
                    # 3. USAGE STRESS SCORE (USS) - Quadratic curve
                    ax3 = fig.add_subplot(2, 3, 3)
                    uss_curve = 1 - x_axis**2  # Theoretical USS curve
                    ax3.plot(x_axis, uss_curve, color='orange', linewidth=2, label='USS = 1 - u²')
                    ax3.plot([metrics['u']], [indicators['USS']], 'ro', markersize=12)
                    ax3.set_title('Usage Stress Score (USS)\nQuadratic Wear', fontweight='bold')
                    ax3.set_xlabel('Usage Metric (u)')
                    ax3.set_ylabel('USS Value')
                    ax3.grid(True, alpha=0.3)
                    ax3.set_xlim(0, 1)
                    ax3.set_ylim(0, 1.05)
                    ax3.text(0.02, 0.95, f'Current: {indicators["USS"]:.3f}', 
                            transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.5))
                    
                    # BOTTOM ROW: Deterioration, Confidence, Valuation
                    
                    # 4. PREDICTIVE DETERIORATION PENALTY (PDP) - Sigmoid curve
                    ax4 = fig.add_subplot(2, 3, 4)
                    pdp_curve = 1 / (1 + np.exp(10 * (x_axis - 0.5)))  # Theoretical PDP curve
                    ax4.plot(x_axis, pdp_curve, color='purple', linewidth=2, label='PDP Sigmoid')
                    ax4.plot([metrics['p']], [indicators['PDP']], 'ro', markersize=12)
                    ax4.set_title('Predictive Deterioration Penalty (PDP)\nSigmoid Aging', fontweight='bold')
                    ax4.set_xlabel('Normalized Age (p)')
                    ax4.set_ylabel('PDP Value')
                    ax4.grid(True, alpha=0.3)
                    ax4.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='θ=0.5')
                    ax4.set_xlim(0, 1)
                    ax4.set_ylim(0, 1.05)
                    ax4.text(0.02, 0.95, f'Current: {indicators["PDP"]:.3f}', 
                            transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
                    
                    # 5. CONFIDENCE INDEX (CI) - History over time
                    ax5 = fig.add_subplot(2, 3, 5)
                    if len(st.session_state.data_log) >= 2:
                        ci_history = [d['technical_indicators']['CI'] for d in st.session_state.data_log[-50:]]
                        ax5.plot(ci_history, 'c-', linewidth=2)
                        ax5.plot([len(ci_history)-1], [ci_history[-1]], 'ro', markersize=12)
                    else:
                        ax5.plot([0, 1], [indicators['CI'], indicators['CI']], 'c-', linewidth=2)
                        ax5.plot([0], [indicators['CI']], 'ro', markersize=12)
                    ax5.set_title('Confidence Index (CI)\nData Quality History', fontweight='bold')
                    ax5.set_xlabel('Time Steps')
                    ax5.set_ylabel('CI Value')
                    ax5.grid(True, alpha=0.3)
                    ax5.set_ylim(0, 1.05)
                    ax5.text(0.02, 0.95, f'Current: {indicators["CI"]:.3f}', 
                            transform=ax5.transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))
                    
                    # 6. RTPMV VALUATION - History over time
                    ax6 = fig.add_subplot(2, 3, 6)
                    if len(st.session_state.data_log) >= 2:
                        rtpmv_history = [d['valuation']['rtpmv'] for d in st.session_state.data_log[-50:]]
                        ax6.plot(rtpmv_history, 'r-', linewidth=2)
                        ax6.plot([len(rtpmv_history)-1], [rtpmv_history[-1]], 'ro', markersize=12)
                        ax6.set_ylim(min(rtpmv_history) * 0.95, max(rtpmv_history) * 1.05)
                    else:
                        current_rtpmv = current['valuation']['rtpmv']
                        ax6.plot([0, 1], [current_rtpmv, current_rtpmv], 'r-', linewidth=2)
                        ax6.plot([0], [current_rtpmv], 'ro', markersize=12)
                        ax6.set_ylim(current_rtpmv * 0.95, current_rtpmv * 1.05)
                    ax6.set_title('Real-Time Property Market Value\nRTPMV ($)', fontweight='bold')
                    ax6.set_xlabel('Time Steps')
                    ax6.set_ylabel('RTPMV ($)')
                    ax6.grid(True, alpha=0.3)
                    # Format y-axis as currency
                    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
                    ax6.text(0.02, 0.95, f'Current: ${current["valuation"]["rtpmv"]:,.0f}', 
                            transform=ax6.transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.5))
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            
            with details_placeholder.container():
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Technical Indicators", "🔐 Verification", 
                                                   "💹 Market Data", "🔬 Sensors"])
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Factor Values")
                        indicators = cycle_data['technical_indicators']
                        st.write(f"**SHF**: {indicators['SHF']:.4f}")
                        st.write(f"**ESF**: {indicators['ESF']:.4f}")
                        st.write(f"**USS**: {indicators['USS']:.4f}")
                        st.write(f"**PDP**: {indicators['PDP']:.4f}")
                        st.write(f"**CI**: {indicators['CI']:.4f}")
                    with col2:
                        st.markdown("### Valuation Breakdown")
                        val = cycle_data['valuation']
                        st.write(f"**Base MV**: ${val['base_market_value']:,.0f}")
                        st.write(f"**Land Value**: ${val['land_value']:,.0f}")
                        st.write(f"**Structure (Current)**: ${val['structure_current']:,.0f}")
                        st.write(f"**Depreciation**: {val['depreciation_percentage']:.2f}%")
                        st.write(f"**RTPMV**: ${val['rtpmv']:,.0f}")
                with tab2:
                    ver = cycle_data['verification']
                    st.markdown(f"**Report ID**: `{ver['report_id']}`")
                    st.markdown(f"**Report Hash**: `{ver['report_hash'][:32]}...`")
                    st.markdown(f"**Token ID**: `{ver['token_id']}`")
                    st.markdown(f"**Token Hash**: `{ver['token_hash'][:32]}...`")
                with tab3:
                    mkt = cycle_data['market']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Pricing")
                        st.write(f"**Listing**: ${mkt['listing_price']:,.0f}")
                        st.write(f"**Bid**: ${mkt['bid_ask']['bid']:,.0f}")
                        st.write(f"**Ask**: ${mkt['bid_ask']['ask']:,.0f}")
                        st.write(f"**Spread**: {mkt['bid_ask']['spread_percentage']:.2f}%")
                    with col2:
                        st.markdown("### Controls")
                        controls = mkt['controls']
                        st.write(f"**Trading**: {'✅ Enabled' if controls['trading_enabled'] else '❌ Disabled'}")
                        if controls['restrictions']:
                            st.warning("\n".join(controls['restrictions']))
                        if controls['requirements']:
                            st.info("\n".join(controls['requirements']))
                with tab4:
                    sensors = cycle_data['sensor_data']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Vibration**: {sensors['vibration']:.3f}")
                        st.write(f"**Strain**: {sensors['strain']:.3f}")
                        st.write(f"**Moisture**: {sensors['moisture']:.3f}")
                        st.write(f"**Temperature**: {sensors['temperature']:.3f}")
                    with col2:
                        st.write(f"**Occupancy**: {sensors['occupancy']:.3f}")
                        st.write(f"**Electrical**: {sensors['electrical_load']:.3f}")
                        st.write(f"**Air Quality**: {sensors['air_quality']:.3f}")
            
            with table_placeholder.container():
                st.markdown("### 📝 Recent Valuation History")
                if len(st.session_state.data_log) > 0:
                    display_data = []
                    for d in st.session_state.data_log[-10:]:
                        display_data.append({
                            'Timestamp': d['timestamp'],
                            'RTPMV': f"${d['valuation']['rtpmv']:,.0f}",
                            'Health': f"{d['valuation']['health_factor']*100:.1f}%",
                            'SHF': f"{d['technical_indicators']['SHF']:.3f}",
                            'ESF': f"{d['technical_indicators']['ESF']:.3f}",
                            'USS': f"{d['technical_indicators']['USS']:.3f}",
                            'CI': f"{d['technical_indicators']['CI']:.3f}",
                            'Token': d['verification']['token_id']
                        })
                    df = pd.DataFrame(display_data)
                    st.dataframe(df, use_container_width=True)
            
            time.sleep(0.5)
    
    else:
        st.info("👆 Click **START** in the sidebar to begin the demonstration")
        
        if len(st.session_state.data_log) > 0:
            st.success(f"✅ Captured {len(st.session_state.data_log)} valuation cycles")
            
            df_export = pd.DataFrame(st.session_state.data_log)
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Data (CSV)",
                data=csv,
                file_name='digital_twin_valuation_data.csv',
                mime='text/csv',
            )
            
            if len(st.session_state.data_log) > 0:
                st.markdown("### 📊 Session Summary")
                col1, col2, col3 = st.columns(3)
                rtpmvs = [d['valuation']['rtpmv'] for d in st.session_state.data_log]
                healths = [d['valuation']['health_factor'] for d in st.session_state.data_log]
                cis = [d['technical_indicators']['CI'] for d in st.session_state.data_log]
                with col1:
                    st.metric("Avg RTPMV", f"${np.mean(rtpmvs):,.0f}")
                    st.metric("Min RTPMV", f"${np.min(rtpmvs):,.0f}")
                    st.metric("Max RTPMV", f"${np.max(rtpmvs):,.0f}")
                with col2:
                    st.metric("Avg Health", f"{np.mean(healths)*100:.1f}%")
                    st.metric("Min Health", f"{np.min(healths)*100:.1f}%")
                    st.metric("Max Health", f"{np.max(healths)*100:.1f}%")
                with col3:
                    st.metric("Avg Confidence", f"{np.mean(cis)*100:.1f}%")
                    st.metric("Min Confidence", f"{np.min(cis)*100:.1f}%")
                    st.metric("Max Confidence", f"{np.max(cis)*100:.1f}%")


if __name__ == "__main__":
    main()
