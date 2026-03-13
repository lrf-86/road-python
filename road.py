"""
道路工程计算后端服务 - Road Engineering Calculation API

基于 FastAPI 框架，提供专业的道路工程计算 RESTful API 接口。
包含纵坡/横坡计算、设计高程计算、路面厚度计算、中线坐标计算等功能。

作者: Bedrock Engineer
版本: 1.0.0
"""

from __future__ import annotations

import math
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# 枚举类型定义
# =============================================================================

class RoadType(str, Enum):
    """道路类型枚举"""
    HIGHWAY = "highway"          # 高速公路
    ARTERIAL = "arterial"        # 主干道
    COLLECTOR = "collector"      # 次干道
    LOCAL = "local"              # 支路
    RURAL = "rural"              # 乡村道路


class PavementType(str, Enum):
    """路面类型枚举"""
    FLEXIBLE = "flexible"        # 柔性路面（沥青混凝土）
    RIGID = "rigid"              # 刚性路面（水泥混凝土）
    COMPOSITE = "composite"      # 复合路面


class TerrainType(str, Enum):
    """地形类型枚举"""
    PLAIN = "plain"              # 平原
    HILLY = "hilly"              # 丘陵
    MOUNTAIN = "mountain"        # 山岭


class CurveType(str, Enum):
    """曲线类型枚举"""
    CIRCULAR = "circular"        # 圆曲线
    SPIRAL = "spiral"            # 缓和曲线
    COMPOUND = "compound"        # 复曲线


# =============================================================================
# 基础数据模型
# =============================================================================

class Point2D(BaseModel):
    """二维坐标点模型"""
    x: float = Field(..., description="X坐标（东坐标）", examples=[100.0])
    y: float = Field(..., description="Y坐标（北坐标）", examples=[200.0])


class Point3D(BaseModel):
    """三维坐标点模型"""
    x: float = Field(..., description="X坐标（东坐标）", examples=[100.0])
    y: float = Field(..., description="Y坐标（北坐标）", examples=[200.0])
    z: float = Field(..., description="Z坐标（高程）", examples=[50.0])


class Station(BaseModel):
    """桩号模型"""
    value: float = Field(..., description="桩号值（米）", ge=0, examples=[1000.0])
    offset: float = Field(0.0, description="偏距（米），左负右正", examples=[-5.0])

    @property
    def formatted(self) -> str:
        """返回格式化的桩号字符串，如 K1+000"""
        km = int(self.value // 1000)
        m = int(self.value % 1000)
        return f"K{km}+{m:03d}"


# =============================================================================
# 请求数据模型
# =============================================================================

class SlopeCalculationRequest(BaseModel):
    """
    纵坡/横坡计算请求模型
    
    用于计算道路的纵向坡度或横向坡度
    """
    start_elevation: float = Field(
        ...,
        description="起点高程（米）",
        examples=[50.0]
    )
    end_elevation: float = Field(
        ...,
        description="终点高程（米）",
        examples=[55.0]
    )
    horizontal_distance: float = Field(
        ...,
        description="水平距离（米），必须大于0",
        gt=0,
        examples=[100.0]
    )
    slope_type: str = Field(
        "longitudinal",
        description="坡度类型: longitudinal(纵坡) 或 cross(横坡)",
        examples=["longitudinal"]
    )

    @field_validator("slope_type")
    @classmethod
    def validate_slope_type(cls, v: str) -> str:
        allowed = ["longitudinal", "cross"]
        if v not in allowed:
            raise ValueError(f"坡度类型必须是以下之一: {allowed}")
        return v


class DesignElevationRequest(BaseModel):
    """
    道路设计高程计算请求模型
    
    根据纵坡设计参数计算任意桩号的设计高程
    """
    start_station: float = Field(
        ...,
        description="起点桩号（米）",
        ge=0,
        examples=[0.0]
    )
    start_elevation: float = Field(
        ...,
        description="起点设计高程（米）",
        examples=[50.0]
    )
    grade: float = Field(
        ...,
        description="纵坡坡度（%），上坡为正，下坡为负",
        examples=[2.5]
    )
    target_station: float = Field(
        ...,
        description="目标桩号（米）",
        ge=0,
        examples=[500.0]
    )
    vertical_curve_length: Optional[float] = Field(
        None,
        description="竖曲线长度（米），用于凸形/凹形竖曲线计算",
        gt=0,
        examples=[80.0]
    )
    curve_type: Optional[str] = Field(
        None,
        description="竖曲线类型: crest(凸形) 或 sag(凹形)",
        examples=["crest"]
    )

    @model_validator(mode="after")
    def validate_curve_params(self) -> "DesignElevationRequest":
        """验证竖曲线参数的一致性"""
        has_curve_length = self.vertical_curve_length is not None
        has_curve_type = self.curve_type is not None
        
        if has_curve_length != has_curve_type:
            raise ValueError("竖曲线长度和竖曲线类型必须同时提供或同时为空")
        
        if has_curve_type and self.curve_type not in ["crest", "sag"]:
            raise ValueError("竖曲线类型必须是 'crest'(凸形) 或 'sag'(凹形)")
        
        return self


class PavementThicknessRequest(BaseModel):
    """
    路面厚度计算请求模型
    
    根据交通荷载和材料参数计算路面结构层厚度
    支持柔性路面和刚性路面设计
    """
    pavement_type: PavementType = Field(
        ...,
        description="路面类型",
        examples=[PavementType.FLEXIBLE]
    )
    design_esal: float = Field(
        ...,
        description="设计年限内标准轴载累计作用次数(ESALs)",
        gt=0,
        examples=[1000000.0]
    )
    subgrade_cbr: float = Field(
        ...,
        description="路基加州承载比 CBR (%)",
        gt=0,
        le=100,
        examples=[8.0]
    )
    reliability: float = Field(
        90.0,
        description="设计可靠度(%)",
        ge=50,
        le=99,
        examples=[90.0]
    )
    resilient_modulus: Optional[float] = Field(
        None,
        description="路基回弹模量(MPa)，如不提供则根据CBR估算",
        gt=0,
        examples=[50.0]
    )
    concrete_flexural_strength: Optional[float] = Field(
        None,
        description="混凝土弯拉强度(MPa)，刚性路面必填",
        gt=0,
        examples=[4.5]
    )
    concrete_elastic_modulus: Optional[float] = Field(
        None,
        description="混凝土弹性模量(MPa)，刚性路面必填",
        gt=0,
        examples=[30000.0]
    )
    k_value: Optional[float] = Field(
        None,
        description="地基反应模量(MPa/m)，刚性路面使用",
        gt=0,
        examples=[50.0]
    )

    @model_validator(mode="after")
    def validate_rigid_params(self) -> "PavementThicknessRequest":
        """验证刚性路面参数"""
        if self.pavement_type == PavementType.RIGID:
            if self.concrete_flexural_strength is None:
                raise ValueError("刚性路面必须提供混凝土弯拉强度")
            if self.concrete_elastic_modulus is None:
                raise ValueError("刚性路面必须提供混凝土弹性模量")
        return self


class CenterlineCoordinateRequest(BaseModel):
    """
    道路中线坐标计算请求模型
    
    根据交点坐标、曲线半径等参数计算道路中线上任意桩号的坐标
    """
    start_point: Point2D = Field(
        ...,
        description="起点坐标"
    )
    end_point: Point2D = Field(
        ...,
        description="终点坐标（用于计算方位角）"
    )
    curve_radius: float = Field(
        ...,
        description="圆曲线半径（米）",
        gt=0,
        examples=[500.0]
    )
    spiral_length: Optional[float] = Field(
        None,
        description="缓和曲线长度（米），如为0或不提供则只有圆曲线",
        ge=0,
        examples=[60.0]
    )
    deflection_angle: float = Field(
        ...,
        description="偏角（度），左偏为负，右偏为正",
        ge=-180,
        le=180,
        examples=[45.0]
    )
    target_station: float = Field(
        ...,
        description="目标桩号（米）",
        ge=0,
        examples=[300.0]
    )
    station_interval: float = Field(
        20.0,
        description="桩号间隔（米），用于批量计算",
        gt=0,
        examples=[20.0]
    )


class BatchCoordinateRequest(BaseModel):
    """
    批量坐标计算请求模型
    
    根据多段路线参数批量计算道路中线坐标
    """
    segments: List[CenterlineCoordinateRequest] = Field(
        ...,
        description="路线分段参数列表",
        min_length=1
    )


class CrossSectionRequest(BaseModel):
    """
    横断面计算请求模型
    
    计算道路横断面上的设计高程和坐标
    """
    centerline_elevation: float = Field(
        ...,
        description="中线设计高程（米）",
        examples=[50.0]
    )
    cross_slope: float = Field(
        ...,
        description="横坡坡度（%），正值为单向坡，负值为双向坡",
        examples=[2.0]
    )
    road_width: float = Field(
        ...,
        description="路面宽度（米）",
        gt=0,
        examples=[7.5]
    )
    shoulder_width: float = Field(
        0.0,
        description="路肩宽度（米）",
        ge=0,
        examples=[0.75]
    )
    shoulder_slope: float = Field(
        3.0,
        description="路肩横坡（%）",
        examples=[3.0]
    )
    offset_points: List[float] = Field(
        default_factory=list,
        description="需要计算的偏距点列表（米），左负右正",
        examples=[[-7.5, -3.75, 0, 3.75, 7.5]]
    )


# =============================================================================
# 响应数据模型
# =============================================================================

class SlopeCalculationResponse(BaseModel):
    """纵坡/横坡计算响应模型"""
    slope_percentage: float = Field(..., description="坡度百分比(%)")
    slope_ratio: str = Field(..., description="坡比表示（如 1:20）")
    slope_angle: float = Field(..., description="坡度角（度）")
    elevation_difference: float = Field(..., description="高差（米）")
    slope_type: str = Field(..., description="坡度类型")
    is_uphill: bool = Field(..., description="是否为上坡/内向坡")


class DesignElevationResponse(BaseModel):
    """设计高程计算响应模型"""
    target_station: float = Field(..., description="目标桩号（米）")
    formatted_station: str = Field(..., description="格式化桩号")
    design_elevation: float = Field(..., description="设计高程（米）")
    grade: float = Field(..., description="纵坡坡度(%)")
    is_on_curve: bool = Field(..., description="是否在竖曲线上")
    curve_elevation_correction: Optional[float] = Field(None, description="竖曲线高程改正值（米）")
    tangent_elevation: Optional[float] = Field(None, description="切线高程（米）")


class PavementLayer(BaseModel):
    """路面结构层模型"""
    layer_name: str = Field(..., description="层位名称")
    material: str = Field(..., description="材料类型")
    thickness: float = Field(..., description="厚度（毫米）")
    modulus: Optional[float] = Field(None, description="模量（MPa）")


class PavementThicknessResponse(BaseModel):
    """路面厚度计算响应模型"""
    pavement_type: str = Field(..., description="路面类型")
    total_thickness: float = Field(..., description="路面结构总厚度（毫米）")
    design_esal: float = Field(..., description="设计ESALs")
    subgrade_cbr: float = Field(..., description="路基CBR(%)")
    subgrade_modulus: float = Field(..., description="路基模量(MPa)")
    reliability: float = Field(..., description="设计可靠度(%)")
    layers: List[PavementLayer] = Field(..., description="路面结构层列表")
    structural_number: Optional[float] = Field(None, description="柔性路面结构数SN")
    concrete_thickness: Optional[float] = Field(None, description="刚性路面混凝土板厚（毫米）")
    safety_factor: Optional[float] = Field(None, description="安全系数")


class CoordinateResult(BaseModel):
    """坐标计算结果模型"""
    station: float = Field(..., description="桩号（米）")
    formatted_station: str = Field(..., description="格式化桩号")
    x: float = Field(..., description="X坐标（东坐标）")
    y: float = Field(..., description="Y坐标（北坐标）")
    tangent_azimuth: float = Field(..., description="切线方位角（度）")
    curve_type: str = Field(..., description="所在曲线类型")


class CenterlineCoordinateResponse(BaseModel):
    """中线坐标计算响应模型"""
    start_point: Point2D = Field(..., description="起点坐标")
    end_point: Point2D = Field(..., description="终点坐标")
    curve_radius: float = Field(..., description="圆曲线半径（米）")
    spiral_length: Optional[float] = Field(None, description="缓和曲线长度（米）")
    deflection_angle: float = Field(..., description="偏角（度）")
    curve_length: float = Field(..., description="曲线总长度（米）")
    tangent_length: float = Field(..., description="切线长（米）")
    external_distance: float = Field(..., description="外距（米）")
    coordinates: List[CoordinateResult] = Field(..., description="计算得到的坐标列表")


class CrossSectionPoint(BaseModel):
    """横断面点模型"""
    offset: float = Field(..., description="偏距（米），左负右正")
    elevation: float = Field(..., description="设计高程（米）")
    slope: float = Field(..., description="该点所在位置的坡度(%)")
    position: str = Field(..., description="位置描述")


class CrossSectionResponse(BaseModel):
    """横断面计算响应模型"""
    centerline_elevation: float = Field(..., description="中线设计高程（米）")
    cross_slope: float = Field(..., description="横坡坡度(%)")
    road_width: float = Field(..., description="路面宽度（米）")
    crown_elevation: Optional[float] = Field(None, description="路拱高程（米）")
    points: List[CrossSectionPoint] = Field(..., description="横断面各点数据")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error_code: str = Field(..., description="错误代码")
    message: str = Field(..., description="错误信息")
    details: Optional[dict] = Field(None, description="详细错误信息")


# =============================================================================
# 核心计算类
# =============================================================================

class RoadCalculator:
    """
    道路工程计算器
    
    提供道路工程相关的专业计算功能
    """

    @staticmethod
    def calculate_slope(
        start_elevation: float,
        end_elevation: float,
        horizontal_distance: float,
        slope_type: str = "longitudinal"
    ) -> SlopeCalculationResponse:
        """
        计算道路坡度
        
        Args:
            start_elevation: 起点高程（米）
            end_elevation: 终点高程（米）
            horizontal_distance: 水平距离（米）
            slope_type: 坡度类型（"longitudinal"纵坡 或 "cross"横坡）
        
        Returns:
            SlopeCalculationResponse: 坡度计算结果
        """
        elevation_diff = end_elevation - start_elevation
        slope_percentage = (elevation_diff / horizontal_distance) * 100
        slope_angle = math.degrees(math.atan(elevation_diff / horizontal_distance))
        
        # 计算坡比
        if abs(slope_percentage) > 0:
            ratio = abs(horizontal_distance / elevation_diff) if elevation_diff != 0 else float('inf')
            if ratio != float('inf'):
                slope_ratio = f"1:{ratio:.1f}"
            else:
                slope_ratio = "0:1"
        else:
            slope_ratio = "0:1"
        
        is_uphill = elevation_diff > 0
        
        return SlopeCalculationResponse(
            slope_percentage=round(slope_percentage, 3),
            slope_ratio=slope_ratio,
            slope_angle=round(slope_angle, 4),
            elevation_difference=round(elevation_diff, 3),
            slope_type=slope_type,
            is_uphill=is_uphill
        )

    @staticmethod
    def calculate_design_elevation(
        start_station: float,
        start_elevation: float,
        grade: float,
        target_station: float,
        vertical_curve_length: Optional[float] = None,
        curve_type: Optional[str] = None
    ) -> DesignElevationResponse:
        """
        计算道路设计高程
        
        Args:
            start_station: 起点桩号（米）
            start_elevation: 起点设计高程（米）
            grade: 纵坡坡度（%）
            target_station: 目标桩号（米）
            vertical_curve_length: 竖曲线长度（米）
            curve_type: 竖曲线类型（"crest"凸形 或 "sag"凹形）
        
        Returns:
            DesignElevationResponse: 设计高程计算结果
        """
        distance = target_station - start_station
        
        # 切线高程（不考虑竖曲线）
        tangent_elevation = start_elevation + (grade / 100) * distance
        
        is_on_curve = False
        curve_correction = None
        final_elevation = tangent_elevation
        
        # 竖曲线高程改正计算
        if vertical_curve_length and curve_type and vertical_curve_length > 0:
            half_curve = vertical_curve_length / 2
            
            # 判断目标桩号是否在竖曲线范围内
            if abs(distance) <= half_curve:
                is_on_curve = True
                x = distance  # 距离竖曲线起点的距离
                
                # 竖曲线高程改正值公式: y = x² / (2R)
                # 其中 R = L / (2 * |Δi|)，这里简化为抛物线竖曲线
                curve_correction = (x ** 2) / (2 * vertical_curve_length * 100 / abs(grade))
                
                if curve_type == "crest":
                    # 凸形竖曲线：改正值为负
                    curve_correction = -abs(curve_correction)
                else:
                    # 凹形竖曲线：改正值为正
                    curve_correction = abs(curve_correction)
                
                final_elevation = tangent_elevation + curve_correction
        
        formatted_station = f"K{int(target_station // 1000)}+{int(target_station % 1000):03d}"
        
        return DesignElevationResponse(
            target_station=round(target_station, 3),
            formatted_station=formatted_station,
            design_elevation=round(final_elevation, 3),
            grade=round(grade, 3),
            is_on_curve=is_on_curve,
            curve_elevation_correction=round(curve_correction, 4) if curve_correction else None,
            tangent_elevation=round(tangent_elevation, 3) if is_on_curve else None
        )

    @staticmethod
    def calculate_pavement_thickness(
        request: PavementThicknessRequest
    ) -> PavementThicknessResponse:
        """
        计算路面结构厚度
        
        根据AASHTO设计方法计算柔性路面或刚性路面厚度
        
        Args:
            request: 路面厚度计算请求参数
        
        Returns:
            PavementThicknessResponse: 路面厚度计算结果
        """
        # 估算路基回弹模量（根据CBR）
        if request.resilient_modulus:
            mr = request.resilient_modulus
        else:
            # 经验公式: Mr (MPa) ≈ 10 * CBR
            mr = 10 * request.subgrade_cbr
        
        layers = []
        
        if request.pavement_type == PavementType.FLEXIBLE:
            # 柔性路面设计（基于AASHTO简化方法）
            return RoadCalculator._calculate_flexible_pavement(request, mr, layers)
        else:
            # 刚性路面设计
            return RoadCalculator._calculate_rigid_pavement(request, mr, layers)

    @staticmethod
    def _calculate_flexible_pavement(
        request: PavementThicknessRequest,
        mr: float,
        layers: List[PavementLayer]
    ) -> PavementThicknessResponse:
        """计算柔性路面厚度"""
        
        # 计算结构数SN（简化AASHTO方法）
        # SN = a1*D1 + a2*D2*m2 + a3*D3*m3
        # 这里使用简化公式估算
        
        # 根据ESALs确定结构数
        if request.design_esal < 50000:
            sn = 1.5
        elif request.design_esal < 150000:
            sn = 2.0
        elif request.design_esal < 500000:
            sn = 2.5
        elif request.design_esal < 2000000:
            sn = 3.0
        elif request.design_esal < 7000000:
            sn = 3.5
        elif request.design_esal < 20000000:
            sn = 4.0
        else:
            sn = 4.5
        
        # 根据可靠度调整
        reliability_factor = 1.0 + (request.reliability - 90) * 0.01
        sn *= reliability_factor
        
        # 根据路基模量调整
        modulus_factor = math.sqrt(50 / mr) if mr > 0 else 1.0
        sn *= max(0.8, min(1.2, modulus_factor))
        
        # 分配各层厚度（典型结构）
        # 面层 - 沥青混凝土
        surface_coefficient = 0.44
        surface_thickness = max(40, (sn * 0.4) / surface_coefficient * 25.4)  # mm
        surface_thickness = round(surface_thickness / 10) * 10  # 取整到10mm
        
        layers.append(PavementLayer(
            layer_name="面层",
            material="沥青混凝土(AC-13)",
            thickness=int(surface_thickness),
            modulus=1400
        ))
        
        # 基层 - 水泥稳定碎石
        base_coefficient = 0.14
        base_thickness = max(150, (sn * 0.35) / base_coefficient * 25.4)  # mm
        base_thickness = round(base_thickness / 10) * 10
        
        layers.append(PavementLayer(
            layer_name="基层",
            material="水泥稳定碎石",
            thickness=int(base_thickness),
            modulus=1500
        ))
        
        # 底基层 - 级配碎石
        subbase_coefficient = 0.11
        subbase_thickness = max(150, (sn * 0.25) / subbase_coefficient * 25.4)  # mm
        subbase_thickness = round(subbase_thickness / 10) * 10
        
        layers.append(PavementLayer(
            layer_name="底基层",
            material="级配碎石",
            thickness=int(subbase_thickness),
            modulus=200
        ))
        
        total_thickness = sum(layer.thickness for layer in layers)
        
        return PavementThicknessResponse(
            pavement_type="柔性路面(沥青混凝土)",
            total_thickness=total_thickness,
            design_esal=request.design_esal,
            subgrade_cbr=request.subgrade_cbr,
            subgrade_modulus=round(mr, 2),
            reliability=request.reliability,
            layers=layers,
            structural_number=round(sn, 2)
        )

    @staticmethod
    def _calculate_rigid_pavement(
        request: PavementThicknessRequest,
        mr: float,
        layers: List[PavementLayer]
    ) -> PavementThicknessResponse:
        """计算刚性路面厚度"""
        
        # 获取参数
        sc = request.concrete_flexural_strength  # MPa
        ec = request.concrete_elastic_modulus    # MPa
        k = request.k_value if request.k_value else mr * 10  # MPa/m
        
        # AASHTO刚性路面设计简化公式
        # log(W18) = ... (复杂公式，这里使用简化估算)
        
        # 根据ESALs估算板厚
        if request.design_esal < 50000:
            thickness = 180
        elif request.design_esal < 200000:
            thickness = 200
        elif request.design_esal < 700000:
            thickness = 220
        elif request.design_esal < 2000000:
            thickness = 240
        elif request.design_esal < 7000000:
            thickness = 260
        elif request.design_esal < 20000000:
            thickness = 280
        else:
            thickness = 300
        
        # 根据弯拉强度调整
        strength_factor = 4.5 / sc if sc > 0 else 1.0
        thickness *= max(0.9, min(1.1, strength_factor))
        
        # 根据可靠度调整
        thickness *= 1.0 + (request.reliability - 90) * 0.002
        
        thickness = round(thickness / 10) * 10  # 取整到10mm
        
        # 混凝土面层
        layers.append(PavementLayer(
            layer_name="面层",
            material="水泥混凝土",
            thickness=int(thickness),
            modulus=ec
        ))
        
        # 基层
        base_thickness = 200
        layers.append(PavementLayer(
            layer_name="基层",
            material="水泥稳定碎石",
            thickness=base_thickness,
            modulus=1500
        ))
        
        # 垫层
        subbase_thickness = 150
        layers.append(PavementLayer(
            layer_name="垫层",
            material="级配碎石",
            thickness=subbase_thickness,
            modulus=200
        ))
        
        total_thickness = sum(layer.thickness for layer in layers)
        
        # 计算安全系数（简化）
        safety_factor = round(sc / (thickness / 200), 2)
        
        return PavementThicknessResponse(
            pavement_type="刚性路面(水泥混凝土)",
            total_thickness=total_thickness,
            design_esal=request.design_esal,
            subgrade_cbr=request.subgrade_cbr,
            subgrade_modulus=round(mr, 2),
            reliability=request.reliability,
            layers=layers,
            concrete_thickness=int(thickness),
            safety_factor=safety_factor
        )

    @staticmethod
    def calculate_centerline_coordinates(
        request: CenterlineCoordinateRequest
    ) -> CenterlineCoordinateResponse:
        """
        计算道路中线坐标
        
        根据交点坐标、曲线半径等参数，使用切线支距法或坐标法
        计算道路中线上各桩号的坐标
        
        Args:
            request: 中线坐标计算请求参数
        
        Returns:
            CenterlineCoordinateResponse: 中线坐标计算结果
        """
        x1, y1 = request.start_point.x, request.start_point.y
        x2, y2 = request.end_point.x, request.end_point.y
        r = request.curve_radius
        ls = request.spiral_length if request.spiral_length else 0
        delta = math.radians(request.deflection_angle)
        
        # 计算起点到交点的方位角
        dx = x2 - x1
        dy = y2 - y1
        azimuth_ij = math.atan2(dy, dx)
        
        # 判断转向
        is_left = delta < 0
        delta_abs = abs(delta)
        
        # 计算曲线要素
        if ls > 0:
            # 有缓和曲线
            # 缓和曲线角
            beta0 = ls / (2 * r)
            
            # 缓和曲线内移值和切线增长值
            p = ls**2 / (24 * r) - ls**4 / (2688 * r**3)
            q = ls / 2 - ls**3 / (240 * r**2)
            
            # 圆曲线部分偏角
            delta_c = delta_abs - 2 * beta0
            
            # 曲线总长
            lc = r * delta_c  # 圆曲线长度
            total_curve_length = 2 * ls + lc
            
            # 切线长
            t = (r + p) * math.tan(delta_abs / 2) + q
            
            # 外距
            e = (r + p) / math.cos(delta_abs / 2) - r
        else:
            # 无缓和曲线，纯圆曲线
            beta0 = 0
            p = 0
            q = 0
            delta_c = delta_abs
            total_curve_length = r * delta_abs
            t = r * math.tan(delta_abs / 2)
            e = r * (1 / math.cos(delta_abs / 2) - 1)
        
        # 计算曲线主点坐标
        # 交点坐标
        x_jd = x1 + t * math.cos(azimuth_ij)
        y_jd = y1 + t * math.sin(azimuth_ij)
        
        # 计算终点坐标（切线方向）
        if is_left:
            azimuth_jd_end = azimuth_ij - delta_abs
        else:
            azimuth_jd_end = azimuth_ij + delta_abs
        
        x_end = x_jd + t * math.cos(azimuth_jd_end)
        y_end = y_jd + t * math.sin(azimuth_jd_end)
        
        # 计算目标桩号坐标
        coordinates = []
        
        # 确定计算范围
        stations_to_calc = []
        
        # 添加目标桩号
        if request.target_station <= total_curve_length:
            stations_to_calc.append(request.target_station)
        
        # 按间隔添加桩号
        current_station = 0.0
        while current_station <= total_curve_length:
            if current_station not in stations_to_calc:
                stations_to_calc.append(current_station)
            current_station += request.station_interval
        
        stations_to_calc.sort()
        
        for station in stations_to_calc:
            coord = RoadCalculator._calculate_single_station(
                station, x1, y1, r, ls, delta_abs, beta0, p, q, 
                azimuth_ij, is_left, total_curve_length
            )
            coordinates.append(coord)
        
        return CenterlineCoordinateResponse(
            start_point=request.start_point,
            end_point=Point2D(x=round(x_end, 3), y=round(y_end, 3)),
            curve_radius=r,
            spiral_length=ls if ls > 0 else None,
            deflection_angle=round(request.deflection_angle, 4),
            curve_length=round(total_curve_length, 3),
            tangent_length=round(t, 3),
            external_distance=round(e, 3),
            coordinates=coordinates
        )

    @staticmethod
    def _calculate_single_station(
        station: float,
        x1: float,
        y1: float,
        r: float,
        ls: float,
        delta: float,
        beta0: float,
        p: float,
        q: float,
        azimuth_start: float,
        is_left: bool,
        total_length: float
    ) -> CoordinateResult:
        """计算单个桩号的坐标"""
        
        formatted = f"K{int(station // 1000)}+{int(station % 1000):03d}"
        
        if ls > 0:
            # 有缓和曲线的情况
            if station <= ls:
                # 第一缓和曲线段
                l = station
                # 缓和曲线参数方程
                x_local = l - l**5 / (40 * r**2 * ls**2)
                y_local = l**3 / (6 * r * ls) - l**7 / (336 * r**3 * ls**3)
                curve_type = "第一缓和曲线"
                
                # 计算切线方位角
                beta = l**2 / (2 * r * ls)
                if is_left:
                    tangent_az = azimuth_start - beta
                else:
                    tangent_az = azimuth_start + beta
                    
            elif station <= ls + r * (delta - 2 * beta0):
                # 圆曲线段
                lc = station - ls
                phi = lc / r + beta0
                
                # 圆曲线坐标（相对于HY点）
                x_hy = ls - ls**3 / (40 * r**2) + q
                y_hy = ls**2 / (6 * r) - ls**4 / (336 * r**3) + p
                
                x_local = x_hy + r * math.sin(phi - beta0)
                y_local = y_hy + r * (1 - math.cos(phi - beta0))
                curve_type = "圆曲线"
                
                # 切线方位角
                if is_left:
                    tangent_az = azimuth_start - phi
                else:
                    tangent_az = azimuth_start + phi
                    
            elif station <= total_length:
                # 第二缓和曲线段
                l = total_length - station
                x_local_end = l - l**5 / (40 * r**2 * ls**2)
                y_local_end = l**3 / (6 * r * ls) - l**7 / (336 * r**3 * ls**3)
                
                # 需要转换到起点坐标系
                x_local = total_length - x_local_end  # 简化处理
                y_local = y_local_end
                curve_type = "第二缓和曲线"
                
                # 切线方位角
                beta = l**2 / (2 * r * ls)
                end_az = azimuth_start + (1 if not is_left else -1) * delta
                if is_left:
                    tangent_az = end_az + beta
                else:
                    tangent_az = end_az - beta
            else:
                # 切线延长段
                x_local = station
                y_local = 0
                curve_type = "切线延长"
                tangent_az = azimuth_start
        else:
            # 纯圆曲线
            if station <= total_length:
                phi = station / r
                x_local = r * math.sin(phi)
                y_local = r * (1 - math.cos(phi))
                curve_type = "圆曲线"
                
                if is_left:
                    tangent_az = azimuth_start - phi
                else:
                    tangent_az = azimuth_start + phi
            else:
                x_local = station
                y_local = 0
                curve_type = "切线延长"
                tangent_az = azimuth_start
        
        # 坐标转换到全局坐标系
        if is_left:
            y_local = -y_local
        
        x_global = x1 + x_local * math.cos(azimuth_start) - y_local * math.sin(azimuth_start)
        y_global = y1 + x_local * math.sin(azimuth_start) + y_local * math.cos(azimuth_start)
        
        # 规范化方位角到0-360度
        tangent_az_deg = math.degrees(tangent_az) % 360
        
        return CoordinateResult(
            station=round(station, 3),
            formatted_station=formatted,
            x=round(x_global, 3),
            y=round(y_global, 3),
            tangent_azimuth=round(tangent_az_deg, 4),
            curve_type=curve_type
        )

    @staticmethod
    def calculate_cross_section(
        request: CrossSectionRequest
    ) -> CrossSectionResponse:
        """
        计算道路横断面
        
        根据中线高程、横坡、路面宽度等参数计算横断面上各点的高程
        
        Args:
            request: 横断面计算请求参数
        
        Returns:
            CrossSectionResponse: 横断面计算结果
        """
        points = []
        
        # 确定计算点
        offsets = request.offset_points if request.offset_points else []
        
        # 如果没有指定偏距点，使用默认点
        if not offsets:
            half_width = request.road_width / 2
            offsets = [
                -half_width - request.shoulder_width,  # 左路肩外缘
                -half_width,                            # 左路缘
                0,                                      # 中线
                half_width,                             # 右路缘
                half_width + request.shoulder_width     # 右路肩外缘
            ]
        
        for offset in offsets:
            abs_offset = abs(offset)
            half_width = request.road_width / 2
            
            if abs_offset <= half_width:
                # 在路面范围内
                if request.cross_slope >= 0:
                    # 单向坡（路拱）
                    elevation = request.centerline_elevation + (offset * request.cross_slope / 100)
                else:
                    # 双向坡（人字坡）
                    if offset < 0:
                        elevation = request.centerline_elevation + abs_offset * abs(request.cross_slope) / 100
                    else:
                        elevation = request.centerline_elevation - abs_offset * abs(request.cross_slope) / 100
                
                slope = request.cross_slope
                position = "行车道"
                
            elif abs_offset <= half_width + request.shoulder_width:
                # 在路肩范围内
                road_edge_offset = half_width if offset > 0 else -half_width
                shoulder_distance = abs(offset - road_edge_offset)
                
                # 路缘处高程
                if request.cross_slope >= 0:
                    road_edge_elev = request.centerline_elevation + (road_edge_offset * request.cross_slope / 100)
                else:
                    road_edge_elev = request.centerline_elevation + half_width * abs(request.cross_slope) / 100
                
                elevation = road_edge_elev + shoulder_distance * request.shoulder_slope / 100
                slope = request.shoulder_slope
                position = "路肩"
                
            else:
                # 边坡范围（简化计算）
                road_edge_offset = half_width if offset > 0 else -half_width
                total_shoulder = half_width + request.shoulder_width
                slope_distance = abs_offset - total_shoulder
                
                if request.cross_slope >= 0:
                    road_edge_elev = request.centerline_elevation + (road_edge_offset * request.cross_slope / 100)
                else:
                    road_edge_elev = request.centerline_elevation + half_width * abs(request.cross_slope) / 100
                
                shoulder_elev = road_edge_elev + request.shoulder_width * request.shoulder_slope / 100
                
                # 边坡坡度假设为1:1.5
                side_slope = 66.67  # %
                elevation = shoulder_elev - slope_distance * side_slope / 100
                slope = -side_slope
                position = "边坡"
            
            points.append(CrossSectionPoint(
                offset=round(offset, 3),
                elevation=round(elevation, 3),
                slope=round(slope, 2),
                position=position
            ))
        
        # 计算路拱高程（如果是双向坡）
        crown_elevation = None
        if request.cross_slope < 0:
            crown_elevation = request.centerline_elevation
        
        return CrossSectionResponse(
            centerline_elevation=request.centerline_elevation,
            cross_slope=request.cross_slope,
            road_width=request.road_width,
            crown_elevation=crown_elevation,
            points=points
        )


# =============================================================================
# FastAPI 应用实例
# =============================================================================

app = FastAPI(
    title="道路工程计算 API",
    description="基于 cecode 土建工程库的道路工程计算后端服务，提供纵坡/横坡计算、设计高程计算、路面厚度计算、中线坐标计算等专业功能",
    version="1.0.0",
    contact={
        "name": "Bedrock Engineer",
        "url": "https://github.com/bedrock-engineer/cecode",
    },
    license_info={
        "name": "MIT",
    },
)

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 全局异常处理
# =============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """处理值错误异常"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": "INVALID_PARAMETER",
            "message": str(exc),
            "details": {}
        }
    )


@app.exception_handler(ZeroDivisionError)
async def zero_division_error_handler(request, exc):
    """处理除零错误异常"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": "DIVISION_BY_ZERO",
            "message": "计算过程中发生除零错误，请检查输入参数",
            "details": {}
        }
    )


# =============================================================================
# API 路由
# =============================================================================

@app.get("/", tags=["系统"])
async def root():
    """
    根路径 - 返回API基本信息
    
    Returns:
        包含服务名称、版本和文档链接的基本信息
    """
    return {
        "service": "道路工程计算 API",
        "version": "1.0.0",
        "description": "基于 cecode 土建工程库的道路工程计算后端服务",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


@app.get("/health", tags=["系统"])
async def health_check():
    """
    健康检查接口
    
    Returns:
        服务状态信息
    """
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"
    }


@app.post(
    "/api/v1/road/slope",
    response_model=SlopeCalculationResponse,
    tags=["坡度计算"],
    summary="计算道路纵坡或横坡",
    description="根据起点高程、终点高程和水平距离计算坡度百分比、坡比和坡度角"
)
async def calculate_slope(request: SlopeCalculationRequest):
    """
    道路坡度计算接口
    
    计算道路纵坡或横坡的各项参数：
    - 坡度百分比（%）
    - 坡比（如 1:20）
    - 坡度角（度）
    - 高差（米）
    
    示例请求:
    ```json
    {
        "start_elevation": 50.0,
        "end_elevation": 55.0,
        "horizontal_distance": 100.0,
        "slope_type": "longitudinal"
    }
    ```
    """
    try:
        result = RoadCalculator.calculate_slope(
            start_elevation=request.start_elevation,
            end_elevation=request.end_elevation,
            horizontal_distance=request.horizontal_distance,
            slope_type=request.slope_type
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"坡度计算失败: {str(e)}"
        )


@app.post(
    "/api/v1/road/design-elevation",
    response_model=DesignElevationResponse,
    tags=["高程计算"],
    summary="计算道路设计高程",
    description="根据纵坡设计参数计算任意桩号的设计高程，支持竖曲线计算"
)
async def calculate_design_elevation(request: DesignElevationRequest):
    """
    道路设计高程计算接口
    
    根据起点桩号、起点高程、纵坡坡度计算目标桩号的设计高程。
    支持竖曲线（凸形/凹形）的高程改正计算。
    
    示例请求（直线段）:
    ```json
    {
        "start_station": 0,
        "start_elevation": 50.0,
        "grade": 2.5,
        "target_station": 500
    }
    ```
    
    示例请求（竖曲线段）:
    ```json
    {
        "start_station": 0,
        "start_elevation": 50.0,
        "grade": 2.5,
        "target_station": 40,
        "vertical_curve_length": 80,
        "curve_type": "crest"
    }
    ```
    """
    try:
        result = RoadCalculator.calculate_design_elevation(
            start_station=request.start_station,
            start_elevation=request.start_elevation,
            grade=request.grade,
            target_station=request.target_station,
            vertical_curve_length=request.vertical_curve_length,
            curve_type=request.curve_type
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设计高程计算失败: {str(e)}"
        )


@app.post(
    "/api/v1/road/pavement-thickness",
    response_model=PavementThicknessResponse,
    tags=["路面设计"],
    summary="计算路面结构厚度",
    description="根据交通荷载、路基参数计算柔性路面或刚性路面的结构厚度"
)
async def calculate_pavement_thickness(request: PavementThicknessRequest):
    """
    路面厚度计算接口
    
    基于AASHTO设计方法计算路面结构厚度：
    - 柔性路面：计算面层、基层、底基层厚度
    - 刚性路面：计算混凝土板厚及基层厚度
    
    柔性路面示例请求:
    ```json
    {
        "pavement_type": "flexible",
        "design_esal": 1000000,
        "subgrade_cbr": 8,
        "reliability": 90
    }
    ```
    
    刚性路面示例请求:
    ```json
    {
        "pavement_type": "rigid",
        "design_esal": 5000000,
        "subgrade_cbr": 10,
        "concrete_flexural_strength": 4.5,
        "concrete_elastic_modulus": 30000,
        "reliability": 95
    }
    ```
    """
    try:
        result = RoadCalculator.calculate_pavement_thickness(request)
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"路面厚度计算失败: {str(e)}"
        )


@app.post(
    "/api/v1/road/centerline-coordinates",
    response_model=CenterlineCoordinateResponse,
    tags=["路线计算"],
    summary="计算道路中线坐标",
    description="根据交点坐标、曲线半径、偏角等参数计算道路中线上各桩号的坐标"
)
async def calculate_centerline_coordinates(request: CenterlineCoordinateRequest):
    """
    道路中线坐标计算接口
    
    使用切线支距法或坐标法计算道路中线上各桩号的坐标：
    - 支持圆曲线和缓和曲线
    - 支持左偏和右偏
    - 返回曲线要素（切线长、曲线长、外距等）
    
    示例请求（圆曲线）:
    ```json
    {
        "start_point": {"x": 100, "y": 200},
        "end_point": {"x": 200, "y": 300},
        "curve_radius": 500,
        "deflection_angle": 45,
        "target_station": 300
    }
    ```
    
    示例请求（带缓和曲线）:
    ```json
    {
        "start_point": {"x": 100, "y": 200},
        "end_point": {"x": 200, "y": 300},
        "curve_radius": 500,
        "spiral_length": 60,
        "deflection_angle": 45,
        "target_station": 300
    }
    ```
    """
    try:
        result = RoadCalculator.calculate_centerline_coordinates(request)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"中线坐标计算失败: {str(e)}"
        )


@app.post(
    "/api/v1/road/cross-section",
    response_model=CrossSectionResponse,
    tags=["横断面计算"],
    summary="计算道路横断面",
    description="根据中线高程、横坡、路面宽度等参数计算横断面上各点的高程"
)
async def calculate_cross_section(request: CrossSectionRequest):
    """
    道路横断面计算接口
    
    计算道路横断面上各特征点的高程：
    - 中线高程
    - 路缘高程
    - 路肩高程
    - 边坡高程
    
    支持单向坡（路拱）和双向坡（人字坡）
    
    示例请求:
    ```json
    {
        "centerline_elevation": 50.0,
        "cross_slope": 2.0,
        "road_width": 7.5,
        "shoulder_width": 0.75,
        "shoulder_slope": 3.0,
        "offset_points": [-7.5, -3.75, 0, 3.75, 7.5]
    }
    ```
    """
    try:
        result = RoadCalculator.calculate_cross_section(request)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"横断面计算失败: {str(e)}"
        )


@app.get(
    "/api/v1/road/grade-check",
    tags=["规范检查"],
    summary="检查纵坡是否符合规范",
    description="根据道路类型检查纵坡是否满足规范要求"
)
async def check_grade_compliance(
    grade: float = Query(..., description="纵坡坡度(%)", ge=-10, le=10),
    road_type: RoadType = Query(..., description="道路类型"),
    terrain: TerrainType = Query(TerrainType.PLAIN, description="地形类型")
):
    """
    纵坡规范检查接口
    
    检查给定纵坡是否符合《公路路线设计规范》要求
    
    Args:
        grade: 纵坡坡度(%)
        road_type: 道路类型
        terrain: 地形类型
    
    Returns:
        规范检查结果，包括是否符合规范、最大允许坡度等
    """
    # 根据规范确定最大纵坡（简化版）
    max_grades = {
        (RoadType.HIGHWAY, TerrainType.PLAIN): 3.0,
        (RoadType.HIGHWAY, TerrainType.HILLY): 4.0,
        (RoadType.HIGHWAY, TerrainType.MOUNTAIN): 5.0,
        (RoadType.ARTERIAL, TerrainType.PLAIN): 4.0,
        (RoadType.ARTERIAL, TerrainType.HILLY): 5.0,
        (RoadType.ARTERIAL, TerrainType.MOUNTAIN): 6.0,
        (RoadType.COLLECTOR, TerrainType.PLAIN): 5.0,
        (RoadType.COLLECTOR, TerrainType.HILLY): 6.0,
        (RoadType.COLLECTOR, TerrainType.MOUNTAIN): 7.0,
        (RoadType.LOCAL, TerrainType.PLAIN): 6.0,
        (RoadType.LOCAL, TerrainType.HILLY): 7.0,
        (RoadType.LOCAL, TerrainType.MOUNTAIN): 8.0,
        (RoadType.RURAL, TerrainType.PLAIN): 6.0,
        (RoadType.RURAL, TerrainType.HILLY): 7.0,
        (RoadType.RURAL, TerrainType.MOUNTAIN): 8.0,
    }
    
    key = (road_type, terrain)
    max_grade = max_grades.get(key, 6.0)
    
    abs_grade = abs(grade)
    is_compliant = abs_grade <= max_grade
    
    return {
        "grade": grade,
        "road_type": road_type.value,
        "terrain": terrain.value,
        "is_compliant": is_compliant,
        "max_allowed_grade": max_grade,
        "message": "符合规范要求" if is_compliant else f"纵坡超限，最大允许{max_grade}%"
    }


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "road:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
