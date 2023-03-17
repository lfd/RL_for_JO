SELECT * FROM comp_cast_type AS cct2, movie_keyword AS mk, complete_cast AS cc, comp_cast_type AS cct1 WHERE cct1.kind = 'cast' AND cct2.kind LIKE '%complete%' AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND cct1.id = cc.subject_id AND cc.subject_id = cct1.id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;