from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
import math
import numpy as np

app = FastAPI(
    title="道路工程计算后端服务",
    description="基于 cecode 土建工程库开发的道路工程专业计算 RESTful API 服务",
    version="1.0.0"
)


class SlopeRequest(BaseModel):
    """纵坡/横坡计算请求模型"""
    point1_elevation: float = Field(description="起点高程 (m)")
    point2_elevation: float = Field(description="终点高程 (m)")
    distance: float = Field(description="两点间水平距离 (m)", gt=0)
    slope_type: str = Field(
        description="坡度类型: 'longitudinal' (纵坡) 或 'transverse' (横坡)",
        default="longitudinal"
    )


class SlopeResponse(BaseModel):
    """纵坡/横坡计算响应模型"""
    slope_ratio: float = Field(description="坡度比 (%)")
    slope_angle: float = Field(description="坡度角 (度)")
    elevation_difference: float = Field(description="高程差 (m)")
    slope_type: str = Field(description="坡度类型")


class DesignElevationPoint(BaseModel):
    """变坡点模型"""
    station: float = Field(description="桩号 (m)")
    elevation: float = Field(description="高程 (m)")
    radius: Optional[float] = Field(description="竖曲线半径 (m), 仅变坡点需要", default=None)


class DesignElevationRequest(BaseModel):
    """设计高程计算请求模型"""
    target_station: float = Field(description="目标桩号 (m)")
    vip_points: List[DesignElevationPoint] = Field(description="变坡点列表")


class DesignElevationResponse(BaseModel):
    """设计高程计算响应模型"""
    target_station: float = Field(description="目标桩号 (m)")
    design_elevation: float = Field(description="设计高程 (m)")
    is_on_vertical_curve: bool = Field(description="是否在竖曲线上")
    vertical_curve_info: Optional[str] = Field(description="竖曲线信息")


class PavementThicknessRequest(BaseModel):
    """路面厚度计算请求模型"""
    traffic_volume: float = Field(description="累计当量轴次 (万次)", gt=0)
    subgrade_modulus: float = Field(description="土基回弹模量 (MPa)", gt=0)
    structural_coefficient: float = Field(description="路面结构层系数", gt=0)
    reliability_level: float = Field(description="可靠度系数", default=1.0)
    design_life: int = Field(description="设计使用年限 (年)", default=15)


class PavementThicknessResponse(BaseModel):
    """路面厚度计算响应模型"""
    total_thickness: float = Field(description="路面总厚度 (cm)")
    surface_thickness: float = Field(description="面层厚度 (cm)")
    base_thickness: float = Field(description="基层厚度 (cm)")
    subbase_thickness: float = Field(description="底基层厚度 (cm)")


class RoadAlignmentPoint(BaseModel):
    """道路中线点模型"""
    station: float = Field(description="桩号 (m)")
    x: float = Field(description="X坐标 (m)")
    y: float = Field(description="Y坐标 (m)")


class CurveElement(BaseModel):
    """平曲线元素模型"""
    curve_type: str = Field(description="曲线类型: 'circular' (圆曲线) 或 'spiral' (缓和曲线)")
    start_station: float = Field(description="曲线起点桩号 (m)")
    end_station: float = Field(description="曲线终点桩号 (m)")
    radius: Optional[float] = Field(description="圆曲线半径 (m)")
    length: Optional[float] = Field(description="缓和曲线长度 (m)")
    deflection_angle: Optional[float] = Field(description="偏角 (度)")
    intersection_x: Optional[float] = Field(description="交点X坐标 (m)")
    intersection_y: Optional[float] = Field(description="交点Y坐标 (m)")


class CenterlineCoordinateRequest(BaseModel):
    """道路中线坐标计算请求模型"""
    target_station: float = Field(description="目标桩号 (m)")
    start_point: RoadAlignmentPoint = Field(description="路线起点")
    start_bearing: float = Field(description="起点方位角 (度)")
    curve_elements: List[CurveElement] = Field(description="平曲线元素列表", default=[])


class CenterlineCoordinateResponse(BaseModel):
    """道路中线坐标计算响应模型"""
    target_station: float = Field(description="目标桩号 (m)")
    x: float = Field(description="X坐标 (m)")
    y: float = Field(description="Y坐标 (m)")
    bearing: float = Field(description="方位角 (度)")
    is_on_curve: bool = Field(description="是否在曲线上")


def calculate_slope(elevation1: float, elevation2: float, distance: float) -> Tuple[float, float, float]:
    """
    计算坡度
    返回: (坡度比%, 坡度角°, 高程差)
    """
    elevation_diff = elevation2 - elevation1
    slope_ratio = (elevation_diff / distance) * 100
    slope_angle = math.degrees(math.atan(elevation_diff / distance))
    return slope_ratio, slope_angle, elevation_diff


def calculate_vertical_curve(vip_prev: DesignElevationPoint, 
                            vip_curr: DesignElevationPoint, 
                            vip_next: DesignElevationPoint,
                            target_station: float) -> Tuple[float, str]:
    """
    计算竖曲线上的设计高程
    返回: (设计高程, 竖曲线信息)
    """
    i1 = (vip_curr.elevation - vip_prev.elevation) / (vip_curr.station - vip_prev.station)
    i2 = (vip_next.elevation - vip_curr.elevation) / (vip_next.station - vip_curr.station)
    
    w = abs(i2 - i1)
    r = vip_curr.radius
    l = w * r
    
    bvc_station = vip_curr.station - l / 2
    evc_station = vip_curr.station + l / 2
    
    if target_station < bvc_station or target_station > evc_station:
        if target_station < vip_curr.station:
            elev = vip_prev.elevation + i1 * (target_station - vip_prev.station)
        else:
            elev = vip_curr.elevation + i2 * (target_station - vip_curr.station)
        return elev, "不在竖曲线上"
    
    x = target_station - bvc_station
    y0 = vip_prev.elevation + i1 * (bvc_station - vip_prev.station)
    
    if (i2 - i1) > 0:
        y = y0 + i1 * x + (x ** 2) / (2 * r)
        curve_type = "凹形竖曲线"
    else:
        y = y0 + i1 * x - (x ** 2) / (2 * r)
        curve_type = "凸形竖曲线"
    
    info = f"{curve_type}, R={r}m, L={l:.2f}m, BVC={bvc_station:.3f}, EVC={evc_station:.3f}"
    return y, info


def calculate_design_elevation(target_station: float, 
                              vip_points: List[DesignElevationPoint]) -> Tuple[float, bool, str]:
    """
    计算设计高程
    返回: (设计高程, 是否在竖曲线上, 竖曲线信息)
    """
    sorted_vips = sorted(vip_points, key=lambda p: p.station)
    
    if target_station <= sorted_vips[0].station:
        i = (sorted_vips[1].elevation - sorted_vips[0].elevation) / (sorted_vips[1].station - sorted_vips[0].station)
        elev = sorted_vips[0].elevation + i * (target_station - sorted_vips[0].station)
        return elev, False, "直线段(起点前)"
    
    if target_station >= sorted_vips[-1].station:
        n = len(sorted_vips)
        i = (sorted_vips[-1].elevation - sorted_vips[-2].elevation) / (sorted_vips[-1].station - sorted_vips[-2].station)
        elev = sorted_vips[-1].elevation + i * (target_station - sorted_vips[-1].station)
        return elev, False, "直线段(终点后)"
    
    for i in range(len(sorted_vips) - 1):
        curr_vip = sorted_vips[i]
        next_vip = sorted_vips[i + 1]
        
        if curr_vip.station <= target_station <= next_vip.station:
            if curr_vip.radius is not None and i > 0:
                elev, info = calculate_vertical_curve(sorted_vips[i-1], curr_vip, next_vip, target_station)
                is_on_curve = "竖曲线" in info
                return elev, is_on_curve, info
            elif next_vip.radius is not None and i + 2 < len(sorted_vips):
                elev, info = calculate_vertical_curve(curr_vip, next_vip, sorted_vips[i+2], target_station)
                is_on_curve = "竖曲线" in info
                return elev, is_on_curve, info
            else:
                slope = (next_vip.elevation - curr_vip.elevation) / (next_vip.station - curr_vip.station)
                elev = curr_vip.elevation + slope * (target_station - curr_vip.station)
                return elev, False, "直线段"
    
    raise ValueError("无法计算设计高程")


def calculate_pavement_thickness(request: PavementThicknessRequest) -> dict:
    """
    路面厚度计算 (基于 AASHTO 1993 经验公式)
    """
    w18 = request.traffic_volume
    mr = request.subgrade_modulus
    zr = request.reliability_level
    sn = request.structural_coefficient
    
    log_w18 = math.log10(w18 * 10000)
    mr_psi = mr * 145.038 
    
    sn = 0.08 * (log_w18 ** 2.5) * (mr_psi / 1000) ** 0.2
    
    surface_thickness = sn * 0.25 / 0.44
    base_thickness = sn * 0.35 / 0.14
    subbase_thickness = sn * 0.40 / 0.08
    
    total = surface_thickness + base_thickness + subbase_thickness
    
    return {
        "total_thickness": round(total, 1),
        "surface_thickness": round(surface_thickness, 1),
        "base_thickness": round(base_thickness, 1),
        "subbase_thickness": round(subbase_thickness, 1)
    }


def calculate_centerline_coordinate(request: CenterlineCoordinateRequest) -> dict:
    """
    计算道路中线坐标
    """
    target_sta = request.target_station
    curr_x = request.start_point.x
    curr_y = request.start_point.y
    curr_sta = request.start_point.station
    curr_bearing = math.radians(request.start_bearing)
    
    if target_sta < curr_sta:
        dist = curr_sta - target_sta
        x = curr_x - dist * math.sin(curr_bearing)
        y = curr_y - dist * math.cos(curr_bearing)
        return {
            "target_station": target_sta,
            "x": round(x, 3),
            "y": round(y, 3),
            "bearing": round(math.degrees(curr_bearing), 4),
            "is_on_curve": False
        }
    
    sorted_curves = sorted(request.curve_elements, key=lambda c: c.start_station)
    
    for curve in sorted_curves:
        if curr_sta >= curve.end_station:
            continue
        
        if target_sta <= curve.start_station:
            dist = target_sta - curr_sta
            x = curr_x + dist * math.sin(curr_bearing)
            y = curr_y + dist * math.cos(curr_bearing)
            return {
                "target_station": target_sta,
                "x": round(x, 3),
                "y": round(y, 3),
                "bearing": round(math.degrees(curr_bearing), 4),
                "is_on_curve": False
            }
        
        if curr_sta < curve.start_station:
            dist = curve.start_station - curr_sta
            curr_x += dist * math.sin(curr_bearing)
            curr_y += dist * math.cos(curr_bearing)
            curr_sta = curve.start_station
        
        if curve.curve_type == "circular":
            r = curve.radius
            defl_angle = math.radians(curve.deflection_angle or 0)
            l_curve = r * abs(defl_angle)
            
            if target_sta > curve.end_station:
                dist_on_curve = curve.end_station - curr_sta
            else:
                dist_on_curve = target_sta - curr_sta
            
            theta = dist_on_curve / r
            if defl_angle < 0:
                theta = -theta
            
            chord = 2 * r * math.sin(theta / 2)
            chord_bearing = curr_bearing + theta / 2
            
            delta_x = chord * math.sin(chord_bearing)
            delta_y = chord * math.cos(chord_bearing)
            
            if target_sta <= curve.end_station:
                return {
                    "target_station": target_sta,
                    "x": round(curr_x + delta_x, 3),
                    "y": round(curr_y + delta_y, 3),
                    "bearing": round(math.degrees(curr_bearing + theta), 4),
                    "is_on_curve": True
                }
            
            curr_x += delta_x
            curr_y += delta_y
            curr_bearing += theta
            curr_sta = curve.end_station
        
        elif curve.curve_type == "spiral":
            ls = curve.length or 0
            r = curve.radius or 1000
            
            if target_sta > curve.end_station:
                dist_on_curve = curve.end_station - curr_sta
            else:
                dist_on_curve = target_sta - curr_sta
            
            l = dist_on_curve
            a = math.sqrt(r * ls)
            phi = (l ** 2) / (2 * r * ls)
            
            x_spiral = l * (1 - (l ** 2) / (40 * r ** 2) + (l ** 4) / (3456 * r ** 4))
            y_spiral = (l ** 3) / (6 * r * ls) * (1 - (l ** 2) / (56 * r ** 2) + (l ** 4) / (7040 * r ** 4))
            
            rot_x = x_spiral * math.cos(curr_bearing) - y_spiral * math.sin(curr_bearing)
            rot_y = x_spiral * math.sin(curr_bearing) + y_spiral * math.cos(curr_bearing)
            
            if target_sta <= curve.end_station:
                return {
                    "target_station": target_sta,
                    "x": round(curr_x + rot_y, 3),
                    "y": round(curr_y + rot_x, 3),
                    "bearing": round(math.degrees(curr_bearing + phi), 4),
                    "is_on_curve": True
                }
            
            curr_x += rot_y
            curr_y += rot_x
            curr_bearing += phi
            curr_sta = curve.end_station
    
    dist = target_sta - curr_sta
    x = curr_x + dist * math.sin(curr_bearing)
    y = curr_y + dist * math.cos(curr_bearing)
    
    return {
        "target_station": target_sta,
        "x": round(x, 3),
        "y": round(y, 3),
        "bearing": round(math.degrees(curr_bearing), 4),
        "is_on_curve": False
    }


@app.post("/api/road/slope/calculate", response_model=SlopeResponse, 
          summary="计算道路纵坡/横坡",
          description="根据两点高程和水平距离计算坡度比和坡度角")
async def calculate_road_slope(request: SlopeRequest):
    try:
        slope_ratio, slope_angle, elev_diff = calculate_slope(
            request.point1_elevation,
            request.point2_elevation,
            request.distance
        )
        return SlopeResponse(
            slope_ratio=round(slope_ratio, 4),
            slope_angle=round(slope_angle, 4),
            elevation_difference=round(elev_diff, 4),
            slope_type=request.slope_type
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"坡度计算失败: {str(e)}")


@app.post("/api/road/design-elevation/calculate", response_model=DesignElevationResponse,
          summary="计算道路设计高程",
          description="根据变坡点信息计算指定桩号的设计高程")
async def calculate_road_design_elevation(request: DesignElevationRequest):
    try:
        if len(request.vip_points) < 2:
            raise HTTPException(status_code=400, detail="至少需要2个变坡点")
        
        elev, is_on_curve, info = calculate_design_elevation(
            request.target_station,
            request.vip_points
        )
        return DesignElevationResponse(
            target_station=request.target_station,
            design_elevation=round(elev, 4),
            is_on_vertical_curve=is_on_curve,
            vertical_curve_info=info if is_on_curve else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"设计高程计算失败: {str(e)}")


@app.post("/api/road/pavement-thickness/calculate", response_model=PavementThicknessResponse,
          summary="计算路面厚度",
          description="根据交通量、土基模量等参数计算路面各结构层厚度")
async def calculate_road_pavement_thickness(request: PavementThicknessRequest):
    try:
        result = calculate_pavement_thickness(request)
        return PavementThicknessResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"路面厚度计算失败: {str(e)}")


@app.post("/api/road/centerline-coordinate/calculate", response_model=CenterlineCoordinateResponse,
          summary="计算道路中线坐标",
          description="根据路线起点、方位角和平曲线元素计算指定桩号的坐标")
async def calculate_road_centerline_coordinate(request: CenterlineCoordinateRequest):
    try:
        result = calculate_centerline_coordinate(request)
        return CenterlineCoordinateResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"中线坐标计算失败: {str(e)}")


@app.get("/api/road/info", summary="获取服务信息")
async def get_service_info():
    return {
        "service": "道路工程计算后端服务",
        "version": "1.0.0",
        "based_on": "cecode 土建工程库",
        "available_apis": [
            "/api/road/slope/calculate - 纵坡/横坡计算",
            "/api/road/design-elevation/calculate - 设计高程计算",
            "/api/road/pavement-thickness/calculate - 路面厚度计算",
            "/api/road/centerline-coordinate/calculate - 中线坐标计算"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print("道路工程计算后端服务启动中...")
    print("访问 http://localhost:8000/docs 查看API文档")
    uvicorn.run(app, host="0.0.0.0", port=8000)
