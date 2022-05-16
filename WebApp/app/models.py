from django.db import models
import geocoder


# Create your models here.
class Data(models.Model):
    location = models.CharField(max_length=100, null=True)
    value = models.FloatField(default=0)
    latitude = models.FloatField(default=0)
    longitude = models.FloatField(default=0)

    class Meta:
        verbose_name_plural = 'Data'

    def save(self, *args, **kwargs):
        self.latitude = geocoder.osm(self.location).lat
        self.longitude = geocoder.osm(self.location).lng
        return super().save(*args, **kwargs)

    def __str__(self):
        return self.location
