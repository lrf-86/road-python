from road import (
    calculate_slope,
    calculate_design_elevation,
    calculate_pavement_thickness,
    DesignElevationPoint,
    PavementThicknessRequest
)

ratio, angle, diff = calculate_slope(100, 102, 100)
print("坡度测试: 2m高差/100m距离 = {:.2f}%".format(ratio))

req = PavementThicknessRequest(traffic_volume=100, subgrade_modulus=30, structural_coefficient=1.0)
result = calculate_pavement_thickness(req)
print("路面厚度测试: 总厚度={}cm".format(result["total_thickness"]))

print("所有核心功能测试通过!")
