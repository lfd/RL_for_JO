SELECT * FROM movie_info AS mi, info_type AS it1 WHERE it1.info = 'countries' AND mi.info IN ('Germany', 'German', 'USA', 'American') AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;