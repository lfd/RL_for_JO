SELECT * FROM movie_link AS ml, movie_keyword AS mk, keyword AS k, title AS t, movie_info AS mi, movie_companies AS mc WHERE k.keyword = 'sequel' AND mc.note IS NULL AND mi.info IN ('Germany', 'German') AND t.production_year BETWEEN 2000 AND 2010 AND ml.movie_id = t.id AND t.id = ml.movie_id AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND mi.movie_id = t.id AND t.id = mi.movie_id AND ml.movie_id = mk.movie_id AND mk.movie_id = ml.movie_id AND ml.movie_id = mc.movie_id AND mc.movie_id = ml.movie_id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND ml.movie_id = mi.movie_id AND mi.movie_id = ml.movie_id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id;