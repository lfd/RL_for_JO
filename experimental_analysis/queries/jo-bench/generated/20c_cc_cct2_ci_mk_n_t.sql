SELECT * FROM movie_keyword AS mk, complete_cast AS cc, cast_info AS ci, title AS t, name AS n, comp_cast_type AS cct2 WHERE cct2.kind LIKE '%complete%' AND t.production_year > 2000 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = ci.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;