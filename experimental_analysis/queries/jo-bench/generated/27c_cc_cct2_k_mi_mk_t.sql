SELECT * FROM comp_cast_type AS cct2, complete_cast AS cc, movie_keyword AS mk, title AS t, keyword AS k, movie_info AS mi WHERE cct2.kind LIKE 'complete%' AND k.keyword = 'sequel' AND mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish', 'Denish', 'Norwegian', 'German', 'English') AND t.production_year BETWEEN 1950 AND 2010 AND t.id = mk.movie_id AND mk.movie_id = t.id AND mk.keyword_id = k.id AND k.id = mk.keyword_id AND mi.movie_id = t.id AND t.id = mi.movie_id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND mk.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mi.movie_id = cc.movie_id AND cc.movie_id = mi.movie_id;