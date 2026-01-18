import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- 1. 数据采集与模拟 (Data Simulation) ---

class DataSimulator:
    """模拟数据生成器"""
    
    def __init__(self):
        self.industries = ['制造加工', '餐饮商超', '仓储物流', '办公服务']
        # 预设行业用电画像 (基准负荷 kW, 峰谷占比)
        self.industry_profiles = {
            '制造加工': {'base_load': 500, 'peak_ratio': 0.6, 'profile': 'stable_high'},
            '餐饮商超': {'base_load': 150, 'peak_ratio': 0.8, 'profile': 'dual_peak'},
            '仓储物流': {'base_load': 80, 'peak_ratio': 0.3, 'profile': 'flat'},
            '办公服务': {'base_load': 200, 'peak_ratio': 0.7, 'profile': 'day_high'}
        }

    def generate_new_businesses(self, num=5):
        """模拟采集新增工商户数据"""
        data = []
        for _ in range(num):
            industry = random.choice(self.industries)
            data.append({
                'company_name': f"模拟企业_{random.randint(1000, 9999)}",
                'industry': industry,
                'registered_capital': random.randint(50, 1000), # 万元
                'scale': random.choice(['S', 'M', 'L']),
                'reg_date': datetime.now().strftime('%Y-%m-%d')
            })
        return pd.DataFrame(data)

    def generate_realtime_power_data(self, hour):
        """模拟某一时刻的电力数据"""
        # 峰谷电价模拟 (简化版：8-22为峰，其余为谷)
        is_peak = 8 <= hour <= 22
        grid_price = 1.2 if is_peak else 0.4
        
        return {
            'timestamp': datetime.now().replace(hour=hour, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S'),
            'hour': hour,
            'is_peak': is_peak,
            'grid_price': grid_price, # 电网电价
            'pv_output': max(0, np.sin((hour - 6) * np.pi / 12) * 1000) if 6 <= hour <= 18 else 0, # 光伏出力模拟
            'grid_load': random.uniform(2000, 5000) + (1000 if is_peak else 0), # 区域总负荷
            'storage_level': random.uniform(20, 90) # 当前储能电量 %
        }

# --- 2. 数据处理与预测 (Processing & Prediction) ---

class DataProcessor:
    """数据处理核心模块"""
    
    def __init__(self, simulator):
        self.simulator = simulator

    def predict_load(self, business_df):
        """基于行业画像预测负荷"""
        predictions = []
        for _, row in business_df.iterrows():
            profile = self.simulator.industry_profiles.get(row['industry'], {})
            # 简单逻辑：注册资本 * 行业基准系数 (仅作演示)
            scale_factor = row['registered_capital'] / 100
            predicted_load = profile.get('base_load', 100) * scale_factor
            
            predictions.append({
                'company_name': row['company_name'],
                'industry': row['industry'],
                'predicted_peak_load': predicted_load,
                'load_profile_type': profile.get('profile')
            })
        return pd.DataFrame(predictions)

# --- 3. 调度决策 (Scheduling Decision) ---

class Scheduler:
    """调度决策核心模块"""
    
    def __init__(self):
        self.storage_capacity = 2000 # kWh
        self.min_soc = 20 # %
        self.max_soc = 90 # %

    def make_decision(self, power_status, predicted_load_sum):
        """
        生成调度指令
        核心逻辑：削峰填谷
        """
        decision = {
            'action': 'HOLD',
            'storage_power': 0, # +充电, -放电
            'grid_purchase': 0,
            'reason': ''
        }
        
        net_load = power_status['grid_load'] - power_status['pv_output']
        current_soc = power_status['storage_level']
        
        if power_status['is_peak']:
            # 峰段逻辑：优先放电，减少购电
            if current_soc > self.min_soc:
                decision['action'] = 'DISCHARGE'
                decision['storage_power'] = -min(500, (current_soc - self.min_soc) / 100 * self.storage_capacity)
                decision['reason'] = '峰段高价，储能放电削峰'
            else:
                decision['reason'] = '峰段但电量不足，停止放电'
        else:
            # 谷段逻辑：低价充电
            if current_soc < self.max_soc:
                decision['action'] = 'CHARGE'
                decision['storage_power'] = min(500, (self.max_soc - current_soc) / 100 * self.storage_capacity)
                decision['reason'] = '谷段低价，储能充电填谷'
            else:
                decision['reason'] = '谷段但电量已满，停止充电'
                
        # 计算最终需要从电网购买的电量
        decision['grid_purchase'] = max(0, net_load + decision['storage_power'])
        return decision

# --- 4. 成本核算 (Cost Analysis) ---

class CostAnalyzer:
    def calculate_margin(self, decision, power_status):
        """简单的边际收益计算"""
        cost = decision['grid_purchase'] * power_status['grid_price']
        # 假设售电价格固定或有一定溢价
        sales_price = power_status['grid_price'] * 1.1 
        revenue = (decision['grid_purchase'] - decision['storage_power']) * sales_price # 粗略计算
        
        return {
            'cost': round(cost, 2),
            'revenue': round(revenue, 2),
            'margin': round(revenue - cost, 2)
        }

# --- Main Execution Flow ---

def run_demo():
    print("=== 虚拟电厂调度系统 POC 演示 ===")
    
    # 1. 初始化
    sim = DataSimulator()
    processor = DataProcessor(sim)
    scheduler = Scheduler()
    analyzer = CostAnalyzer()
    
    # 2. 获取新增用户并预测
    print("\n[Step 1] 采集新增工商户数据...")
    new_businesses = sim.generate_new_businesses(3)
    print(new_businesses[['company_name', 'industry', 'registered_capital']])
    
    print("\n[Step 2] 负荷预测...")
    load_preds = processor.predict_load(new_businesses)
    total_new_load = load_preds['predicted_peak_load'].sum()
    print(load_preds)
    print(f"新增预测总负荷: {total_new_load:.2f} kW")
    
    # 3. 模拟不同时段的调度
    hours_to_sim = [4, 10, 14, 20] # 凌晨(谷), 上午(峰), 下午(平/峰), 晚上(峰)
    
    print("\n[Step 3] 实时调度决策模拟...")
    for h in hours_to_sim:
        print(f"\n--- 时间: {h}:00 ---")
        power_status = sim.generate_realtime_power_data(h)
        print(f"电网状态: 电价={power_status['grid_price']}元, 负荷={power_status['grid_load']:.1f}kW, 光伏={power_status['pv_output']:.1f}kW, SOC={power_status['storage_level']:.1f}%")
        
        decision = scheduler.make_decision(power_status, total_new_load)
        print(f"调度指令: {decision['action']} | 储能功率: {decision['storage_power']:.1f}kW | 理由: {decision['reason']}")
        
        financials = analyzer.calculate_margin(decision, power_status)
        print(f"经济测算: 成本={financials['cost']}, 预计营收={financials['revenue']}, 毛利={financials['margin']}")

if __name__ == "__main__":
    run_demo()
