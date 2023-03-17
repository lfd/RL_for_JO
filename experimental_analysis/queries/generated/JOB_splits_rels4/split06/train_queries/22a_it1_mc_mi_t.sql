SELECT * FROM title AS t, movie_companies AS mc, movie_info AS mi, info_type AS it1 WHERE it1.info = 'countries' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND mi.info IN ('Germany', 'German', 'USA', 'American') AND t.production_year > 2008 AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND it1.id = mi.info_type_id AND mi.info_type_id = it1.id;