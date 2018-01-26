#!/usr/bin/env python
# -*-coding:utf-8 -*-
# created by 'zjh' on 18-1-26

from shapely.geometry import mapping, Polygon
import fiona

def geojson2shp(geojson, shpdst, id):
    '''

    :param geojson: the geojson format of a polygon
    :param shpdst: the path of the shapefile
    :param id: the id property
    :return: no return, just save the shapefile into the shpdst
    '''
    # an example Shapely geometry
    coordinates = geojson['coordinates']
    poly = Polygon(coordinates)

    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }

    # Write a new Shapefile
    with fiona.open(shpdst, 'w', 'ESRI Shapefile', schema) as c:
        ## If there are multiple geometries, put the "for" loop here
        c.write({
            'geometry': mapping(poly),
            'properties': {'id': id},
        })