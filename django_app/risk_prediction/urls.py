from django.urls import path

from .views import (
    predict_risk,
    get_product_data
)


from django.urls import path

from .views import (
    predict_risk,
    get_product_data
)


urlpatterns = [

    path(
        "",
        predict_risk,
        name="predict_risk"
    ),

    path(
        "product-data/",
        get_product_data,
        name="get_product_data"
    ),

]