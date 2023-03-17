SELECT * FROM company_type AS ct, kind_type AS kt, keyword AS k, movie_keyword AS mk, title AS t, complete_cast AS cc, comp_cast_type AS cct1, movie_companies AS mc WHERE cct1.kind = 'complete+verified' AND kt.kind IN ('movie', 'tv movie', 'video movie', 'video game') AND t.production_year > 1990 AND kt.id = t.kind_id AND t.kind_id = kt.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND t.id = cc.movie_id AND cc.movie_id = t.id AND mk.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = cc.movie_id AND cc.movie_id = mk.movie_id AND mc.movie_id = cc.movie_id AND cc.movie_id = mc.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id AND ct.id = mc.company_type_id AND mc.company_type_id = ct.id AND cct1.id = cc.status_id AND cc.status_id = cct1.id;