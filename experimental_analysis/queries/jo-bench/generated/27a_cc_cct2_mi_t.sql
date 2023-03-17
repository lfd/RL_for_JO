SELECT * FROM movie_info AS mi, comp_cast_type AS cct2, complete_cast AS cc, title AS t WHERE cct2.kind = 'complete' AND mi.info IN ('Sweden', 'Germany', 'Swedish', 'German') AND t.production_year BETWEEN 1950 AND 2000 AND mi.movie_id = t.id AND t.id = mi.movie_id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id;