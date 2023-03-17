SELECT * FROM comp_cast_type AS cct2, movie_companies AS mc, complete_cast AS cc, movie_keyword AS mk, title AS t WHERE cct2.kind != 'complete+verified' AND mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%' AND t.production_year > 2005 AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;