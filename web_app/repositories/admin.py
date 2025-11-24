from django.contrib import admin

import repositories.models as request_models

admin.site.register(request_models.ModelStatus)
admin.site.register(request_models.Stock)
admin.site.register(request_models.StockData)