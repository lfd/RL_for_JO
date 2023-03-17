SELECT * FROM comp_cast_type AS cct1, kind_type AS kt, keyword AS k, title AS t, movie_keyword AS mk, complete_cast AS cc, movie_companies AS mc WHERE cct1.kind = 'complete+verified' AND kt.kind IN ('movie') AND t.production_year > 2000 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND cct1.id = cc.status_id AND cc.status_id = cct1.id;