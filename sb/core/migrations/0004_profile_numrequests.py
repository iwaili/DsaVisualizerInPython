# Generated by Django 5.0.4 on 2024-05-07 14:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0003_alter_profile_options'),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='numrequests',
            field=models.IntegerField(default=0),
        ),
    ]
