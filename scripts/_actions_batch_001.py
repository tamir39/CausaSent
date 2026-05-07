"""Action diversification — batch 1.

Targets the top ~50 most repeated (aspect, sentiment, action) triples in
data/gold/{train,val,test}.json. Each key maps to 3–4 diverse rewrites;
`scripts/diversify_actions.py` rotates them round-robin across occurrences
so e.g. 172× 'Duy trì thái độ nhân viên' becomes 4 different actions
~43 times each. This kills exact-duplicate dominance without hurting intent.

Coverage: ~960 / 2273 annotations (42 %).
"""

OVERRIDES: dict[tuple[str, str, str], list[str]] = {
    # ─── customer_service / positive ──────────────────────────────────────
    ("customer_service", "positive", "Duy trì thái độ nhân viên"): [
        "Tiếp tục đào tạo nhân viên thân thiện",
        "Khen thưởng nhân viên phục vụ tốt",
        "Giữ vững phong cách phục vụ chuyên nghiệp",
        "Phát huy tinh thần phục vụ hiện tại",
    ],
    ("customer_service", "positive", "Duy trì thái độ phục vụ"): [
        "Phát huy phong cách phục vụ niềm nở",
        "Tiếp tục huấn luyện kỹ năng giao tiếp",
        "Khích lệ tinh thần phục vụ tận tình",
        "Giữ vững chuẩn mực phục vụ khách",
    ],
    ("customer_service", "positive", "Duy trì thái độ lễ tân"): [
        "Phát huy phong cách lễ tân chuyên nghiệp",
        "Đào tạo nâng cao kỹ năng lễ tân",
        "Khen thưởng lễ tân phục vụ tốt",
    ],
    ("customer_service", "positive", "Duy trì thái độ chủ nhà"): [
        "Phát huy sự thân thiện của chủ nhà",
        "Khích lệ tinh thần đón khách của chủ",
        "Tiếp tục giữ phong cách hiếu khách",
    ],
    ("customer_service", "positive", "Duy trì thái độ chủ quán"): [
        "Phát huy sự niềm nở của chủ quán",
        "Khích lệ tinh thần phục vụ của chủ",
        "Giữ vững phong cách thân thiện của chủ",
    ],
    ("customer_service", "positive", "Duy trì thái độ chủ"): [
        "Phát huy sự nhiệt tình của chủ",
        "Khích lệ tinh thần đón khách của chủ",
        "Giữ vững sự thân thiện của chủ",
    ],
    ("customer_service", "positive", "Duy trì tốc độ phục vụ"): [
        "Tối ưu quy trình phục vụ nhanh",
        "Phát huy hiệu suất phục vụ hiện tại",
        "Giữ vững tốc độ lên món",
    ],
    ("customer_service", "positive", "Duy trì sự chuyên nghiệp"): [
        "Phát huy năng lực chuyên nghiệp của đội ngũ",
        "Đào tạo nâng cao tính chuyên nghiệp",
        "Khen thưởng nhân viên chuyên nghiệp",
    ],

    # ─── customer_service / negative ──────────────────────────────────────
    ("customer_service", "negative", "Đào tạo lại nhân viên"): [
        "Tổ chức khóa huấn luyện kỹ năng phục vụ",
        "Rà soát quy trình tuyển và đào tạo",
        "Đánh giá lại thái độ phục vụ định kỳ",
        "Bổ sung khóa học giao tiếp khách hàng",
    ],
    ("customer_service", "negative", "Đào tạo lại lễ tân"): [
        "Huấn luyện lại quy trình check-in lễ tân",
        "Bổ sung kỹ năng giao tiếp cho lễ tân",
        "Rà soát thái độ lễ tân định kỳ",
    ],

    # ─── product_quality / positive ────────────────────────────────────────
    ("product_quality", "positive", "Duy trì độ sạch"): [
        "Giữ vững quy trình vệ sinh hiện tại",
        "Phát huy tiêu chuẩn sạch sẽ",
        "Tiếp tục giám sát chất lượng vệ sinh",
        "Khen thưởng đội ngũ giữ phòng",
    ],
    ("product_quality", "positive", "Duy trì chất lượng"): [
        "Giữ vững tiêu chuẩn hiện tại",
        "Tiếp tục đầu tư cho chất lượng",
        "Phát huy điểm mạnh chất lượng",
    ],
    ("product_quality", "positive", "Duy trì chất lượng món"): [
        "Giữ công thức chế biến hiện tại",
        "Tiếp tục kiểm soát nguyên liệu đầu vào",
        "Phát huy hương vị đặc trưng",
        "Khen thưởng đầu bếp",
    ],
    ("product_quality", "positive", "Duy trì bữa sáng"): [
        "Giữ vững thực đơn bữa sáng hiện tại",
        "Tiếp tục đầu tư cho bữa sáng",
        "Phát huy chất lượng bữa sáng",
    ],
    ("product_quality", "positive", "Duy trì chất lượng phòng"): [
        "Giữ vững tiêu chuẩn phòng nghỉ",
        "Bảo trì định kỳ tiện nghi phòng",
        "Phát huy chất lượng nội thất phòng",
    ],
    ("product_quality", "positive", "Duy trì chất lượng dịch vụ"): [
        "Giữ vững tiêu chuẩn dịch vụ",
        "Phát huy điểm mạnh dịch vụ hiện tại",
        "Tiếp tục giám sát chất lượng dịch vụ",
    ],
    ("product_quality", "positive", "Duy trì khẩu phần"): [
        "Giữ định lượng món ăn hiện tại",
        "Phát huy ưu thế khẩu phần lớn",
        "Tiếp tục đảm bảo no đủ cho khách",
    ],
    ("product_quality", "positive", "Duy trì khẩu phần lớn"): [
        "Giữ định lượng đầy đặn cho mỗi phần",
        "Phát huy điểm mạnh khẩu phần dồi dào",
        "Tiếp tục cung cấp khẩu phần lớn",
    ],
    ("product_quality", "positive", "Duy trì chất lượng bữa sáng"): [
        "Giữ vững thực đơn bữa sáng phong phú",
        "Phát huy chất lượng bữa sáng đa dạng",
        "Tiếp tục đầu tư cho buffet sáng",
    ],
    ("product_quality", "positive", "Duy trì bữa sáng đa dạng"): [
        "Giữ thực đơn sáng phong phú",
        "Phát huy sự đa dạng buffet sáng",
        "Tiếp tục mở rộng lựa chọn bữa sáng",
    ],
    ("product_quality", "positive", "Duy trì chất lượng tổng thể"): [
        "Giữ vững trải nghiệm tổng thể",
        "Phát huy điểm mạnh toàn diện",
        "Tiếp tục duy trì tiêu chuẩn cao",
    ],
    ("product_quality", "positive", "Duy trì giường"): [
        "Bảo trì định kỳ chất lượng giường",
        "Giữ vững tiêu chuẩn nệm và giường",
        "Phát huy sự thoải mái của giường",
    ],
    ("product_quality", "positive", "Duy trì nước dùng"): [
        "Giữ công thức nước dùng đặc trưng",
        "Phát huy hương vị nước dùng",
        "Tiếp tục kiểm soát chất lượng nước dùng",
    ],
    ("product_quality", "positive", "Duy trì độ sạch và tiện nghi"): [
        "Bảo trì định kỳ tiện nghi và vệ sinh",
        "Giữ tiêu chuẩn sạch sẽ và đầy đủ tiện nghi",
        "Phát huy ưu thế phòng sạch tiện nghi",
    ],

    # ─── product_quality / negative ────────────────────────────────────────
    ("product_quality", "negative", "Đa dạng bữa sáng"): [
        "Mở rộng thực đơn bữa sáng",
        "Bổ sung lựa chọn cho buffet sáng",
        "Cải thiện sự phong phú của bữa sáng",
        "Thêm món Á / Âu vào bữa sáng",
    ],

    # ─── appearance / positive ────────────────────────────────────────────
    ("appearance", "positive", "Quảng bá vị trí"): [
        "Nhấn mạnh ưu thế vị trí trong marketing",
        "Đưa vị trí vào điểm bán nổi bật",
        "Quảng cáo tận dụng lợi thế địa điểm",
    ],
    ("appearance", "positive", "Quảng bá vị trí trung tâm"): [
        "Nhấn mạnh vị trí trung tâm khi marketing",
        "Đưa vị trí trung tâm vào USP",
        "Quảng cáo lợi thế gần trung tâm",
    ],
    ("appearance", "positive", "Quảng bá vị trí gần biển"): [
        "Nhấn mạnh ưu thế gần biển khi marketing",
        "Đưa vị trí ven biển vào điểm bán",
        "Quảng cáo lợi thế sát bãi biển",
    ],
    ("appearance", "positive", "Quảng bá vị trí thuận tiện"): [
        "Nhấn mạnh sự thuận tiện đi lại",
        "Đưa vị trí thuận tiện vào marketing",
        "Quảng cáo ưu thế giao thông",
    ],
    ("appearance", "positive", "Quảng bá vị trí và view"): [
        "Nhấn mạnh vị trí kèm tầm nhìn đẹp",
        "Đưa view và địa điểm vào USP",
        "Quảng cáo lợi thế view đẹp gần trung tâm",
    ],
    ("appearance", "positive", "Duy trì thiết kế đẹp"): [
        "Phát huy điểm nhấn thiết kế",
        "Tiếp tục đầu tư cho mặt bằng",
        "Giữ vững phong cách thiết kế hiện tại",
    ],
    ("appearance", "positive", "Duy trì thiết kế"): [
        "Phát huy phong cách thiết kế hiện tại",
        "Tiếp tục đầu tư cho ngoại thất",
        "Giữ vững định hướng thiết kế",
    ],
    ("appearance", "positive", "Duy trì không gian rộng"): [
        "Giữ vững bố cục không gian rộng rãi",
        "Phát huy ưu thế không gian thoáng",
        "Tiếp tục tận dụng diện tích lớn",
    ],
    ("appearance", "positive", "Duy trì không gian thoáng"): [
        "Giữ vững sự thoáng đãng của không gian",
        "Phát huy ưu thế không gian thoáng mát",
        "Tiếp tục tối ưu thông gió",
    ],

    # ─── appearance / negative ────────────────────────────────────────────
    ("appearance", "negative", "Mở rộng phòng"): [
        "Cân nhắc tăng diện tích phòng",
        "Thiết kế lại bố cục để phòng rộng hơn",
        "Bổ sung loại phòng có diện tích lớn",
    ],
    ("appearance", "negative", "Cải thiện thiết kế"): [
        "Đầu tư làm mới thiết kế",
        "Cập nhật phong cách thiết kế",
        "Thay đổi nội thất cho hiện đại hơn",
    ],
    ("appearance", "negative", "Đổi mới thiết kế iPhone"): [
        "Đầu tư R&D thiết kế dòng iPhone mới",
        "Cập nhật ngôn ngữ thiết kế iPhone",
        "Thay đổi tổng thể kiểu dáng iPhone",
    ],
    ("appearance", "negative", "Mở rộng không gian"): [
        "Cân nhắc tăng diện tích",
        "Thiết kế lại bố cục cho rộng hơn",
        "Tối ưu sắp xếp để tăng không gian",
    ],
    ("appearance", "negative", "Cải thiện chỉ dẫn vị trí"): [
        "Bổ sung biển chỉ dẫn rõ ràng",
        "Cập nhật bản đồ Google chính xác hơn",
        "Hướng dẫn đường đi chi tiết qua website",
    ],

    # ─── price / positive ─────────────────────────────────────────────────
    ("price", "positive", "Giữ giá rẻ"): [
        "Duy trì chính sách giá cạnh tranh",
        "Tiếp tục giữ mức giá hấp dẫn",
        "Phát huy ưu thế giá thấp",
    ],
    ("price", "positive", "Giữ giá hợp lý"): [
        "Duy trì chính sách giá cân đối",
        "Tiếp tục giữ giá phù hợp chất lượng",
        "Phát huy mức giá tương xứng",
    ],
    ("price", "positive", "Giữ mức giá hợp lý"): [
        "Duy trì chính sách giá hài hòa",
        "Tiếp tục cân bằng giá và chất lượng",
        "Phát huy mức giá tương xứng",
    ],
    ("price", "positive", "Giữ mức giá tốt"): [
        "Duy trì lợi thế giá hiện tại",
        "Tiếp tục giữ chính sách giá tốt",
        "Phát huy ưu thế cạnh tranh về giá",
    ],
    ("price", "positive", "Giữ giá phải chăng"): [
        "Duy trì mức giá phổ thông",
        "Tiếp tục cân bằng giá phải chăng",
        "Phát huy ưu thế giá vừa phải",
    ],
    ("price", "positive", "Giữ giá bình dân"): [
        "Duy trì chính sách giá bình dân",
        "Tiếp tục phục vụ phân khúc bình dân",
        "Phát huy ưu thế giá đại chúng",
    ],
    ("price", "positive", "Giữ giá 25k"): [
        "Duy trì mức giá 25k cho món hiện tại",
        "Tiếp tục giữ giá 25k như niêm yết",
    ],
    ("price", "positive", "Giữ giá 30k"): [
        "Duy trì mức giá 30k cho phần ăn",
        "Tiếp tục giữ giá 30k đầy đủ",
    ],

    # ─── price / negative ─────────────────────────────────────────────────
    ("price", "negative", "Cân nhắc giảm giá"): [
        "Rà soát lại chính sách giá",
        "Bổ sung gói giá thấp hơn cho phân khúc nhạy cảm",
        "Đánh giá tương quan giá và chất lượng",
        "Mở thêm chương trình khuyến mãi định kỳ",
    ],

    # ─── price / neutral ──────────────────────────────────────────────────
    ("price", "neutral", "Niêm yết giá rõ"): [
        "Công bố bảng giá minh bạch",
        "Cập nhật giá rõ ràng trên menu",
        "Hiển thị giá đầy đủ tại điểm bán",
    ],

    # ─── usability / negative ─────────────────────────────────────────────
    ("usability", "negative", "Nâng cấp wifi"): [
        "Đầu tư hệ thống wifi mạnh hơn",
        "Tăng băng thông và phủ sóng wifi",
        "Bổ sung router cho khu vực sóng yếu",
    ],
}
