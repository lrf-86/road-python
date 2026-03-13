"""
道路工程计算后端服务
基于 cecode 土建工程库，提供专业的道路工程计算 RESTful API 接口

功能模块：
1. 道路纵坡/横坡计算
2. 道路设计高程计算
3. 路面厚度计算
4. 道路中线坐标计算
"""

from typing import List, Optional, Tuple
from enum import Enum
from math import radians, sin, cos, tan, sqrt, atan, degrees, pi
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
import uvicorn


app = FastAPI(
    title="道路工程计算服务",
    description="基于 cecode 土建工程库的专业道路工程计算 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


class RoadClass(str, Enum):
    """道路等级枚举"""
    EXPRESSWAY = "expressway"
    ARTERIAL = "arterial"
    COLLECTOR = "collector"
    LOCAL = "local"


class PavementType(str, Enum):
    """路面类型枚举"""
    ASPHALT = "asphalt"
    CONCRETE = "concrete"
    COMPOSITE = "composite"


class TerrainType(str, Enum):
    """地形类型枚举"""
    PLAIN = "plain"
    HILLY = "hilly"
    MOUNTAINOUS = "mountainous"


class SlopeDirection(str, Enum):
    """坡向枚举"""
    UPHILL = "uphill"
    DOWNHILL = "downhill"


class GradientRequest(BaseModel):
    """纵坡计算请求模型"""
    start_elevation: float = Field(..., ge=-500, le=9000, description="起点高程
    end_elevation: float = Field(..., ge=-500, le=9000, description="终点高程
    horizontal_distance: float = Field(..., gt=0, description="水平距离
    design_speed: Optional[float] = Field(None, ge=20, le=120, description="设计速度

    @field_validator('end_elevation')
    @classmethod
    def validate_elevations(cls, v, info):
        if 'start_elevation' in info.data:
            elevation_diff = abs(v - info.data['start_elevation'])
            if elevation_diff > 1000:
                raise ValueError('起点与终点高程差不应超过1000米')
        return v


class CrossSlopeRequest(BaseModel):
    """横坡计算请求模型"""
    road_width: float = Field(..., gt=0, le=50, description="道路宽度
    elevation_difference: float = Field(..., description="边缘高程差
    lane_count: int = Field(2, ge=1, le=12, description="车道数")
    has_median: bool = Field(False, description="是否有中央分隔带")

    @field_validator('elevation_difference')
    @classmethod
    def validate_elevation_diff(cls, v):
        if abs(v) > 1.0:
            raise ValueError('横坡高程差一般不超过1米')
        return v


class DesignElevationRequest(BaseModel):
    """设计高程计算请求模型"""
    start_elevation: float = Field(..., description="起点设计高程
    gradient: float = Field(..., ge=-10, le=10, description="纵坡坡度(%)")
    horizontal_distance: float = Field(..., gt=0, description="水平距离
    vertical_curve_radius: Optional[float] = Field(None, gt=0, description="竖曲线半径
    curve_length: Optional[float] = Field(None, gt=0, description="竖曲线长度
    position_on_curve: Optional[float] = Field(None, ge=0, description="曲线上位置点距离

    @field_validator('gradient')
    @classmethod
    def validate_gradient(cls, v):
        if abs(v) > 8:
            raise ValueError('纵坡坡度一般不超过8%')
        return v


class PavementThicknessRequest(BaseModel):
    """路面厚度计算请求模型"""
    road_class: RoadClass = Field(..., description="道路等级")
    pavement_type: PavementType = Field(..., description="路面类型")
    design_axle_load: float = Field(..., gt=0, le=200, description="设计轴载
    design_life_years: int = Field(..., ge=5, le=30, description="设计年限(年)")
    traffic_growth_rate: float = Field(0.05, ge=0, le=0.15, description="交通量增长率")
    initial_traffic: int = Field(..., gt=0, description="初始日交通量(辆/日)")
    subgrade_modulus: float = Field(..., gt=0, le=500, description="路基回弹模量
    terrain_type: TerrainType = Field(TerrainType.PLAIN, description="地形类型")


class CenterlineCoordinateRequest(BaseModel):
    """中线坐标计算请求模型"""
    start_point: Tuple[float, float] = Field(..., description="起点坐标
    azimuth: float = Field(..., ge=0, lt=360, description="起始方位角(度)")
    horizontal_distance: float = Field(..., gt=0, description="水平距离
    curve_radius: Optional[float] = Field(None, gt=0, description="平曲线半径
    deflection_angle: Optional[float] = Field(None, ge=-180, le=180, description="偏角(度)")
    transition_curve_length: Optional[float] = Field(None, ge=0, description="缓和曲线长度


class GradientResponse(BaseModel):
    """纵坡计算响应模型"""
    gradient_percent: float = Field(..., description="纵坡坡度(%)")
    gradient_ratio: str = Field(..., description="纵坡比(1:n)")
    slope_direction: SlopeDirection = Field(..., description="坡向")
    slope_length: float = Field(..., description="坡长
    meets_standard: bool = Field(..., description="是否满足规范要求")
    max_allowed_gradient: float = Field(..., description="允许最大纵坡(%)")
    warning_message: Optional[str] = Field(None, description="警告信息")


class CrossSlopeResponse(BaseModel):
    """横坡计算响应模型"""
    cross_slope_percent: float = Field(..., description="横坡坡度(%)")
    cross_slope_ratio: str = Field(..., description="横坡比(1:n)")
    drainage_adequate: bool = Field(..., description="排水是否满足要求")
    recommended_slope: float = Field(..., description="推荐横坡值(%)")
    meets_standard: bool = Field(..., description="是否满足规范要求")


class DesignElevationResponse(BaseModel):
    """设计高程计算响应模型"""
    end_elevation: float = Field(..., description="终点设计高程
    elevation_change: float = Field(..., description="高程变化量
    curve_correction: Optional[float] = Field(None, description="竖曲线修正值
    final_elevation: float = Field(..., description="最终设计高程
    grade_points: List[dict] = Field(default_factory=list, description="变坡点信息")


class PavementThicknessResponse(BaseModel):
    """路面厚度计算响应模型"""
    total_thickness: float = Field(..., description="路面总厚度
    surface_course: float = Field(..., description="面层厚度
    base_course: float = Field(..., description="基层厚度
    subbase_course: float = Field(..., description="底基层厚度
    design_traffic: float = Field(..., description="设计交通量(累计轴次)")
    structural_number: float = Field(..., description="结构数")
    meets_requirements: bool = Field(..., description="是否满足设计要求")


class CenterlineCoordinateResponse(BaseModel):
    """中线坐标计算响应模型"""
    end_point: Tuple[float, float] = Field(..., description="终点坐标
    intermediate_points: List[dict] = Field(default_factory=list, description="中间点坐标")
    curve_parameters: Optional[dict] = Field(None, description="曲线参数")
    tangent_length: Optional[float] = Field(None, description="切线长
    curve_length: Optional[float] = Field(None, description="曲线长


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error_code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误信息")
    details: Optional[dict] = Field(None, description="详细信息")


class RoadEngineeringCalculator:
    """道路工程计算核心类"""
    
    MAX_GRADIENT_BY_SPEED = {
        120: 3.0,
        100: 4.0,
        80: 5.0,
        60: 6.0,
        40: 7.0,
        30: 8.0,
        20: 9.0
    }
    
    MIN_GRADIENT = 0.3
    
    CROSS_SLOPE_RANGE = {
        'asphalt': (1.5, 2.5),
        'concrete': (1.0, 2.0),
        'composite': (1.5, 2.5)
    }
    
    SURFACE_THICKNESS = {
        'expressway': {'asphalt': 0.18, 'concrete': 0.26, 'composite': 0.20},
        'arterial': {'asphalt': 0.15, 'concrete': 0.24, 'composite': 0.18},
        'collector': {'asphalt': 0.12, 'concrete': 0.22, 'composite': 0.15},
        'local': {'asphalt': 0.08, 'concrete': 0.20, 'composite': 0.12}
    }

    @staticmethod
    def calculate_gradient(
        start_elevation: float,
        end_elevation: float,
        horizontal_distance: float,
        design_speed: Optional[float] = None
    ) -> dict:
        """
        计算道路纵坡
        
        公式: i = (H2 - H1) / L × 100%
        
        Args:
            start_elevation: 起点高程
            end_elevation: 终点高程
            horizontal_distance: 水平距离
            design_speed: 设计速度
            
        Returns:
            包含纵坡计算结果的字典
        """
        elevation_diff = end_elevation - start_elevation
        gradient_percent = (elevation_diff / horizontal_distance) * 100
        slope_length = sqrt(horizontal_distance**2 + elevation_diff**2)
        
        if gradient_percent > 0:
            slope_direction = SlopeDirection.UPHILL
        elif gradient_percent < 0:
            slope_direction = SlopeDirection.DOWNHILL
        else:
            slope_direction = SlopeDirection.UPHILL
        
        abs_gradient = abs(gradient_percent)
        
        max_allowed = 8.0
        meets_standard = True
        warning_message = None
        
        if design_speed:
            for speed, max_grad in sorted(RoadEngineeringCalculator.MAX_GRADIENT_BY_SPEED.items()):
                if design_speed >= speed:
                    max_allowed = max_grad
                    break
        
        if abs_gradient > max_allowed:
            meets_standard = False
            warning_message = f"纵坡{abs_gradient:.2f}%超过允许最大值{max_allowed}%"
        elif abs_gradient < RoadEngineeringCalculator.MIN_GRADIENT and abs_gradient > 0:
            warning_message = f"纵坡{abs_gradient:.2f}%小于最小排水坡度{RoadEngineeringCalculator.MIN_GRADIENT}%"
        
        if abs_gradient > 0:
            gradient_ratio = f"1:{abs(1/(gradient_percent/100)):.0f}"
        else:
            gradient_ratio = "0"
        
        return {
            'gradient_percent': round(gradient_percent, 3),
            'gradient_ratio': gradient_ratio,
            'slope_direction': slope_direction,
            'slope_length': round(slope_length, 3),
            'meets_standard': meets_standard,
            'max_allowed_gradient': max_allowed,
            'warning_message': warning_message
        }

    @staticmethod
    def calculate_cross_slope(
        road_width: float,
        elevation_difference: float,
        lane_count: int,
        has_median: bool,
        pavement_type: PavementType = PavementType.ASPHALT
    ) -> dict:
        """
        计算道路横坡
        
        公式: i = Δh / B × 100%
        
        Args:
            road_width: 道路宽度
            elevation_difference: 边缘高程差
            lane_count: 车道数
            has_median: 是否有中央分隔带
            pavement_type: 路面类型
            
        Returns:
            包含横坡计算结果的字典
        """
        cross_slope_percent = (abs(elevation_difference) / road_width) * 100
        
        recommended_range = RoadEngineeringCalculator.CROSS_SLOPE_RANGE.get(
            pavement_type.value, (1.5, 2.5)
        )
        recommended_slope = (recommended_range[0] + recommended_range[1]) / 2
        
        drainage_adequate = cross_slope_percent >= RoadEngineeringCalculator.MIN_GRADIENT
        meets_standard = recommended_range[0] <= cross_slope_percent <= recommended_range[1]
        
        if cross_slope_percent > 0:
            cross_slope_ratio = f"1:{abs(1/(cross_slope_percent/100)):.0f}"
        else:
            cross_slope_ratio = "0"
        
        return {
            'cross_slope_percent': round(cross_slope_percent, 3),
            'cross_slope_ratio': cross_slope_ratio,
            'drainage_adequate': drainage_adequate,
            'recommended_slope': round(recommended_slope, 2),
            'meets_standard': meets_standard
        }

    @staticmethod
    def calculate_design_elevation(
        start_elevation: float,
        gradient: float,
        horizontal_distance: float,
        vertical_curve_radius: Optional[float] = None,
        curve_length: Optional[float] = None,
        position_on_curve: Optional[float] = None
    ) -> dict:
        """
        计算道路设计高程
        
        直线段: H = H0 + i × L
        竖曲线: H = H0 + i × x ± x²/(2R)
        
        Args:
            start_elevation: 起点设计高程
            gradient: 纵坡坡度(%)
            horizontal_distance: 水平距离
            vertical_curve_radius: 竖曲线半径
            curve_length: 竖曲线长度
            position_on_curve: 曲线上位置点距离
            
        Returns:
            包含设计高程计算结果的字典
        """
        gradient_decimal = gradient / 100
        elevation_change = gradient_decimal * horizontal_distance
        end_elevation = start_elevation + elevation_change
        
        curve_correction = None
        final_elevation = end_elevation
        grade_points = []
        
        if vertical_curve_radius and curve_length and position_on_curve is not None:
            if position_on_curve <= curve_length / 2:
                x = position_on_curve
            else:
                x = curve_length - position_on_curve
            
            curve_correction = (x ** 2) / (2 * vertical_curve_radius)
            
            if gradient > 0:
                final_elevation = end_elevation - curve_correction
            else:
                final_elevation = end_elevation + curve_correction
            
            grade_points.append({
                'position': 'BVC',
                'distance': 0,
                'elevation': round(start_elevation, 3)
            })
            grade_points.append({
                'position': 'PVI',
                'distance': round(curve_length / 2, 3),
                'elevation': round(start_elevation + gradient_decimal * (curve_length / 2), 3)
            })
            grade_points.append({
                'position': 'EVC',
                'distance': round(curve_length, 3),
                'elevation': round(end_elevation, 3)
            })
        
        return {
            'end_elevation': round(end_elevation, 3),
            'elevation_change': round(elevation_change, 3),
            'curve_correction': round(curve_correction, 6) if curve_correction else None,
            'final_elevation': round(final_elevation, 3),
            'grade_points': grade_points
        }

    @staticmethod
    def calculate_pavement_thickness(
        road_class: RoadClass,
        pavement_type: PavementType,
        design_axle_load: float,
        design_life_years: int,
        traffic_growth_rate: float,
        initial_traffic: int,
        subgrade_modulus: float,
        terrain_type: TerrainType
    ) -> dict:
        """
        计算路面厚度
        
        基于AASHTO路面设计方法
        
        Args:
            road_class: 道路等级
            pavement_type: 路面类型
            design_axle_load: 设计轴载
            design_life_years: 设计年限
            traffic_growth_rate: 交通量增长率
            initial_traffic: 初始日交通量
            subgrade_modulus: 路基回弹模量
            terrain_type: 地形类型
            
        Returns:
            包含路面厚度计算结果的字典
        """
        design_traffic = initial_traffic * 365 * design_life_years * (
            (1 + traffic_growth_rate) ** design_life_years - 1
        ) / traffic_growth_rate if traffic_growth_rate > 0 else initial_traffic * 365 * design_life_years
        
        design_traffic = design_traffic * 0.5
        
        terrain_factor = {
            TerrainType.PLAIN: 1.0,
            TerrainType.HILLY: 1.1,
            TerrainType.MOUNTAINOUS: 1.2
        }.get(terrain_type, 1.0)
        
        subgrade_factor = max(0.8, min(1.2, subgrade_modulus / 100))
        
        base_structural_number = 3.5 if road_class == RoadClass.EXPRESSWAY else \
                                 3.0 if road_class == RoadClass.ARTERIAL else \
                                 2.5 if road_class == RoadClass.COLLECTOR else 2.0
        
        structural_number = base_structural_number * terrain_factor / subgrade_factor
        
        surface_thickness = RoadEngineeringCalculator.SURFACE_THICKNESS[road_class.value][pavement_type.value]
        
        if pavement_type == PavementType.ASPHALT:
            base_thickness = 0.20 + (structural_number - 2.0) * 0.05
            subbase_thickness = 0.15 + (design_traffic / 1e7) * 0.05
        elif pavement_type == PavementType.CONCRETE:
            base_thickness = 0.15 + (structural_number - 2.0) * 0.03
            subbase_thickness = 0.12 + (design_traffic / 1e7) * 0.04
        else:
            base_thickness = 0.18 + (structural_number - 2.0) * 0.04
            subbase_thickness = 0.14 + (design_traffic / 1e7) * 0.045
        
        base_thickness = max(0.15, min(0.35, base_thickness))
        subbase_thickness = max(0.10, min(0.30, subbase_thickness))
        
        total_thickness = surface_thickness + base_thickness + subbase_thickness
        
        meets_requirements = total_thickness >= 0.45 if road_class in [RoadClass.EXPRESSWAY, RoadClass.ARTERIAL] else total_thickness >= 0.35
        
        return {
            'total_thickness': round(total_thickness, 3),
            'surface_course': round(surface_thickness, 3),
            'base_course': round(base_thickness, 3),
            'subbase_course': round(subbase_thickness, 3),
            'design_traffic': round(design_traffic, 0),
            'structural_number': round(structural_number, 3),
            'meets_requirements': meets_requirements
        }

    @staticmethod
    def calculate_centerline_coordinate(
        start_point: Tuple[float, float],
        azimuth: float,
        horizontal_distance: float,
        curve_radius: Optional[float] = None,
        deflection_angle: Optional[float] = None,
        transition_curve_length: Optional[float] = None
    ) -> dict:
        """
        计算道路中线坐标
        
        直线段坐标计算:
        X = X0 + L × cos(α)
        Y = Y0 + L × sin(α)
        
        圆曲线坐标计算:
        X = X0 + R × sin(θ)
        Y = Y0 + R × (1 - cos(θ))
        
        Args:
            start_point: 起点坐标
            azimuth: 起始方位角(度)
            horizontal_distance: 水平距离
            curve_radius: 平曲线半径
            deflection_angle: 偏角(度)
            transition_curve_length: 缓和曲线长度
            
        Returns:
            包含中线坐标计算结果的字典
        """
        azimuth_rad = radians(azimuth)
        
        if curve_radius is None or deflection_angle is None:
            end_x = start_point[0] + horizontal_distance * cos(azimuth_rad)
            end_y = start_point[1] + horizontal_distance * sin(azimuth_rad)
            
            return {
                'end_point': (round(end_x, 3), round(end_y, 3)),
                'intermediate_points': [],
                'curve_parameters': None,
                'tangent_length': None,
                'curve_length': None
            }
        
        deflection_rad = radians(abs(deflection_angle))
        
        tangent_length = curve_radius * tan(deflection_rad / 2)
        curve_length = curve_radius * deflection_rad
        
        external_distance = curve_radius * (1 / cos(deflection_rad / 2) - 1)
        
        mid_ordinal = curve_radius * (1 - cos(deflection_rad / 2))
        
        tc_azimuth = azimuth_rad
        bc_x = start_point[0] + tangent_length * cos(tc_azimuth)
        bc_y = start_point[1] + tangent_length * sin(tc_azimuth)
        
        if deflection_angle > 0:
            center_azimuth = azimuth_rad + radians(90)
        else:
            center_azimuth = azimuth_rad - radians(90)
        
        center_x = bc_x + curve_radius * cos(center_azimuth)
        center_y = bc_y + curve_radius * sin(center_azimuth)
        
        intermediate_points = []
        num_points = max(5, int(curve_length / 20))
        
        for i in range(num_points + 1):
            ratio = i / num_points
            angle_on_curve = deflection_rad * ratio
            
            if deflection_angle > 0:
                point_azimuth = azimuth_rad + angle_on_curve
            else:
                point_azimuth = azimuth_rad - angle_on_curve
            
            dist_from_bc = curve_radius * angle_on_curve
            
            chord_length = 2 * curve_radius * sin(angle_on_curve / 2)
            
            if deflection_angle > 0:
                chord_azimuth = azimuth_rad + angle_on_curve / 2
            else:
                chord_azimuth = azimuth_rad - angle_on_curve / 2
            
            point_x = bc_x + chord_length * cos(chord_azimuth)
            point_y = bc_y + chord_length * sin(chord_azimuth)
            
            intermediate_points.append({
                'station': round(ratio * curve_length, 3),
                'x': round(point_x, 3),
                'y': round(point_y, 3),
                'deflection': round(degrees(angle_on_curve), 3)
            })
        
        ec_azimuth = azimuth_rad + deflection_rad if deflection_angle > 0 else azimuth_rad - deflection_rad
        ec_x = bc_x + 2 * curve_radius * sin(deflection_rad / 2) * cos(azimuth_rad + (deflection_rad / 2 if deflection_angle > 0 else -deflection_rad / 2))
        ec_y = bc_y + 2 * curve_radius * sin(deflection_rad / 2) * sin(azimuth_rad + (deflection_rad / 2 if deflection_angle > 0 else -deflection_rad / 2))
        
        end_x = ec_x + tangent_length * cos(ec_azimuth)
        end_y = ec_y + tangent_length * sin(ec_azimuth)
        
        curve_parameters = {
            'radius': curve_radius,
            'deflection_angle': deflection_angle,
            'external_distance': round(external_distance, 3),
            'mid_ordinal': round(mid_ordinal, 3),
            'transition_curve_length': transition_curve_length
        }
        
        return {
            'end_point': (round(end_x, 3), round(end_y, 3)),
            'intermediate_points': intermediate_points,
            'curve_parameters': curve_parameters,
            'tangent_length': round(tangent_length, 3),
            'curve_length': round(curve_length, 3)
        }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            message=str(exc.detail),
            details=None
        ).model_dump()
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """值错误异常处理器"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error_code="INVALID_VALUE",
            message="输入参数值无效",
            details={"error": str(exc)}
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用异常处理器"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="服务器内部错误",
            details={"error": str(exc)}
        ).model_dump()
    )


@app.get("/", tags=["系统"])
async def root():
    """
    服务根路径
    
    返回服务基本信息和可用接口列表
    """
    return {
        "service": "道路工程计算服务",
        "version": "1.0.0",
        "description": "基于 cecode 土建工程库的专业道路工程计算 API",
        "endpoints": {
            "纵坡计算": "/api/v1/gradient",
            "横坡计算": "/api/v1/cross-slope",
            "设计高程计算": "/api/v1/design-elevation",
            "路面厚度计算": "/api/v1/pavement-thickness",
            "中线坐标计算": "/api/v1/centerline-coordinate"
        },
        "documentation": "/docs"
    }


@app.get("/health", tags=["系统"])
async def health_check():
    """
    健康检查接口
    
    用于监控服务运行状态
    """
    return {"status": "healthy", "service": "road-engineering-calculator"}


@app.post(
    "/api/v1/gradient",
    response_model=GradientResponse,
    tags=["道路几何计算"],
    summary="道路纵坡计算",
    description="""
计算道路纵坡坡度及相关参数。

**计算公式**: i = (H₂ - H₁) / L × 100%

**参数说明**:
- start_elevation: 起点高程
- end_elevation: 终点高程
- horizontal_distance: 水平距离
- design_speed: 设计速度，用于确定最大允许纵坡
"""
)
async def calculate_gradient(request: GradientRequest):
    """
    道路纵坡计算接口
    
    根据起终点高程和水平距离计算纵坡坡度，并判断是否满足规范要求。
    """
    try:
        result = RoadEngineeringCalculator.calculate_gradient(
            start_elevation=request.start_elevation,
            end_elevation=request.end_elevation,
            horizontal_distance=request.horizontal_distance,
            design_speed=request.design_speed
        )
        return GradientResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"纵坡计算失败: {str(e)}"
        )


@app.post(
    "/api/v1/cross-slope",
    response_model=CrossSlopeResponse,
    tags=["道路几何计算"],
    summary="道路横坡计算",
    description="""
计算道路横坡坡度及相关参数。

**计算公式**: i = Δh / B × 100%

**参数说明**:
- road_width: 道路宽度
- elevation_difference: 边缘高程差
- lane_count: 车道数
- has_median: 是否有中央分隔带
"""
)
async def calculate_cross_slope(request: CrossSlopeRequest):
    """
    道路横坡计算接口
    
    根据道路宽度和边缘高程差计算横坡坡度，评估排水性能。
    """
    try:
        result = RoadEngineeringCalculator.calculate_cross_slope(
            road_width=request.road_width,
            elevation_difference=request.elevation_difference,
            lane_count=request.lane_count,
            has_median=request.has_median
        )
        return CrossSlopeResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"横坡计算失败: {str(e)}"
        )


@app.post(
    "/api/v1/design-elevation",
    response_model=DesignElevationResponse,
    tags=["道路几何计算"],
    summary="道路设计高程计算",
    description="""
计算道路设计高程。

**直线段公式**: H = H₀ + i × L

**竖曲线公式**: H = H₀ + i × x ± x²/(2R)

**参数说明**:
- start_elevation: 起点设计高程
- gradient: 纵坡坡度(%)
- horizontal_distance: 水平距离
- vertical_curve_radius: 竖曲线半径(可选)
- curve_length: 竖曲线长度(可选)
- position_on_curve: 曲线上位置点距离(可选)
"""
)
async def calculate_design_elevation(request: DesignElevationRequest):
    """
    道路设计高程计算接口
    
    根据起点高程、纵坡和距离计算设计高程，支持竖曲线修正。
    """
    try:
        result = RoadEngineeringCalculator.calculate_design_elevation(
            start_elevation=request.start_elevation,
            gradient=request.gradient,
            horizontal_distance=request.horizontal_distance,
            vertical_curve_radius=request.vertical_curve_radius,
            curve_length=request.curve_length,
            position_on_curve=request.position_on_curve
        )
        return DesignElevationResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设计高程计算失败: {str(e)}"
        )


@app.post(
    "/api/v1/pavement-thickness",
    response_model=PavementThicknessResponse,
    tags=["路面设计"],
    summary="路面厚度计算",
    description="""
计算路面结构层厚度。

基于 AASHTO 路面设计方法，考虑道路等级、路面类型、交通量、路基条件等因素。

**参数说明**:
- road_class: 道路等级
- pavement_type: 路面类型
- design_axle_load: 设计轴载
- design_life_years: 设计年限
- traffic_growth_rate: 交通量增长率
- initial_traffic: 初始日交通量
- subgrade_modulus: 路基回弹模量
- terrain_type: 地形类型
"""
)
async def calculate_pavement_thickness(request: PavementThicknessRequest):
    """
    路面厚度计算接口
    
    根据道路等级、交通量和路基条件计算路面各结构层厚度。
    """
    try:
        result = RoadEngineeringCalculator.calculate_pavement_thickness(
            road_class=request.road_class,
            pavement_type=request.pavement_type,
            design_axle_load=request.design_axle_load,
            design_life_years=request.design_life_years,
            traffic_growth_rate=request.traffic_growth_rate,
            initial_traffic=request.initial_traffic,
            subgrade_modulus=request.subgrade_modulus,
            terrain_type=request.terrain_type
        )
        return PavementThicknessResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"路面厚度计算失败: {str(e)}"
        )


@app.post(
    "/api/v1/centerline-coordinate",
    response_model=CenterlineCoordinateResponse,
    tags=["道路几何计算"],
    summary="道路中线坐标计算",
    description="""
计算道路中线坐标。

**直线段公式**:
- X = X₀ + L × cos(α)
- Y = Y₀ + L × sin(α)

**圆曲线公式**:
- 切线长 T = R × tan(θ/2)
- 曲线长 L = R × θ

**参数说明**:
- start_point: 起点坐标
- azimuth: 起始方位角(度)
- horizontal_distance: 水平距离
- curve_radius: 平曲线半径(可选)
- deflection_angle: 偏角(可选)
- transition_curve_length: 缓和曲线长度(可选)
"""
)
async def calculate_centerline_coordinate(request: CenterlineCoordinateRequest):
    """
    道路中线坐标计算接口
    
    根据起点坐标、方位角和距离计算中线坐标，支持平曲线计算。
    """
    try:
        result = RoadEngineeringCalculator.calculate_centerline_coordinate(
            start_point=request.start_point,
            azimuth=request.azimuth,
            horizontal_distance=request.horizontal_distance,
            curve_radius=request.curve_radius,
            deflection_angle=request.deflection_angle,
            transition_curve_length=request.transition_curve_length
        )
        return CenterlineCoordinateResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"中线坐标计算失败: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "road:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
