SELECT * FROM comp_cast_type AS cct2, complete_cast AS cc, comp_cast_type AS cct1, title AS t, movie_companies AS mc WHERE cct1.kind IN ('cast', 'crew') AND cct2.kind = 'complete' AND mc.note IS NULL AND t.production_year = 1998 AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id;