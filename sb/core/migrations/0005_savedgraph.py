# Generated by Django 5.0.4 on 2024-05-07 18:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0004_profile_numrequests'),
    ]

    operations = [
        migrations.CreateModel(
            name='SavedGraph',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=100)),
                ('requestno', models.CharField(max_length=100)),
                ('ab', models.CharField(max_length=100)),
                ('image', models.ImageField(upload_to='graphs/')),
            ],
        ),
    ]
