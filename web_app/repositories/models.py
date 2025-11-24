from django.db import models


class ModelStatus(models.Model):
    model_name = models.CharField(max_length=100)
    last_trained = models.DateTimeField(auto_now=True)
    accuracy = models.FloatField()
    status = models.CharField(max_length=50)

    def __str__(self):
        output = f"{self.model_name} - Last Trained: {self.last_trained} - Accuracy: {self.accuracy} - Status: {self.status}"

        return output
    

class Stock(models.Model):
    symbol = models.CharField(max_length=10, unique=True)

    def __str__(self):
        return f"{self.symbol}"


class StockData(models.Model):
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    time = models.DateTimeField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    volume = models.BigIntegerField()
    is_predicted = models.BooleanField(default=False)

    class Meta:
        unique_together = ('stock', 'time')
        ordering = ['-time']

    def __str__(self):
        return f"{self.stock.symbol} - {self.time}"

class CeleryTaskRecord(models.Model):
    task_id = models.CharField(max_length=50, unique=True)
    task_name = models.CharField(max_length=100)
    status = models.CharField(max_length=50)
    result = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Task {self.task_name} ({self.task_id}) - Status: {self.status}"